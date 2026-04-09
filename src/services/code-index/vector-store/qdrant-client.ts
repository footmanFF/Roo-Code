import { QdrantClient, Schemas } from "@qdrant/js-client-rest"
import { createHash } from "crypto"
import * as path from "path"
import { v5 as uuidv5 } from "uuid"
import { IVectorStore } from "../interfaces/vector-store"
import { Payload, VectorStoreSearchResult } from "../interfaces"
import { DEFAULT_MAX_SEARCH_RESULTS, DEFAULT_SEARCH_MIN_SCORE, QDRANT_CODE_BLOCK_NAMESPACE } from "../constants"
import { t } from "../../../i18n"

/**
 * IVectorStore 的 Qdrant 实现。
 * 负责集合初始化、向量写入、相似度检索、按文件删除以及索引完成状态标记。
 */
export class QdrantVectorStore implements IVectorStore {
	private readonly vectorSize!: number
	private readonly DISTANCE_METRIC = "Cosine"

	private client: QdrantClient
	private readonly collectionName: string
	private readonly qdrantUrl: string = "http://localhost:6333"
	private readonly workspacePath: string

	/**
	 * 创建 Qdrant 向量存储实例。
	 *
	 * @param workspacePath 工作区路径（用于生成隔离的集合名）
	 * @param url           Qdrant 服务地址
	 * @param vectorSize    向量维度（必须与 embedding 模型一致）
	 * @param apiKey        可选 API Key
	 */
	constructor(workspacePath: string, url: string, vectorSize: number, apiKey?: string) {
		// 先规范化 URL，确保支持 hostname、host:port、完整 URL 等多种输入形式
		const parsedUrl = this.parseQdrantUrl(url)

		// 保存解析后的 URL，供错误提示与诊断使用
		this.qdrantUrl = parsedUrl
		this.workspacePath = workspacePath

		try {
			const urlObj = new URL(parsedUrl)

			// 统一使用 host/port 方式构造客户端，避免 SDK 默认端口导致歧义
			let port: number
			let useHttps: boolean

			if (urlObj.port) {
				// URL 显式携带端口时，直接使用
				port = Number(urlObj.port)
				useHttps = urlObj.protocol === "https:"
			} else {
				// 未显式端口时按协议推断默认端口
				if (urlObj.protocol === "https:") {
					port = 443
					useHttps = true
				} else {
					// http 或其他协议默认 80
					port = 80
					useHttps = false
				}
			}

			this.client = new QdrantClient({
				host: urlObj.hostname,
				https: useHttps,
				port: port,
				prefix: urlObj.pathname === "/" ? undefined : urlObj.pathname.replace(/\/+$/, ""),
				apiKey,
				headers: {
					"User-Agent": "Roo-Code",
				},
			})
		} catch (urlError) {
			// URL 解析失败时降级使用 url 直传模式（兜底）
			// 注意：该模式对 prefix 的处理能力较弱，仅用于容错
			this.client = new QdrantClient({
				url: parsedUrl,
				apiKey,
				headers: {
					"User-Agent": "Roo-Code",
				},
			})
		}

		// 基于 workspacePath 生成稳定且隔离的集合名，避免不同工程冲突
		const hash = createHash("sha256").update(workspacePath).digest("hex")
		this.vectorSize = vectorSize
		this.collectionName = `ws-${hash.substring(0, 16)}`
	}

	/**
	 * 解析并规范化 Qdrant 服务地址。
	 *
	 * 支持输入：
	 * - `localhost`
	 * - `localhost:6333`
	 * - `http://localhost:6333`
	 * - `https://host/prefix`
	 *
	 * @param url 原始 URL 输入
	 * @returns   可用于 QdrantClient 的规范化 URL
	 */
	private parseQdrantUrl(url: string | undefined): string {
		// 空值时回退到默认本地地址
		if (!url || url.trim() === "") {
			return "http://localhost:6333"
		}

		const trimmedUrl = url.trim()

		// 未携带协议时按 hostname 处理并补全协议
		if (!trimmedUrl.startsWith("http://") && !trimmedUrl.startsWith("https://") && !trimmedUrl.includes("://")) {
			return this.parseHostname(trimmedUrl)
		}

		try {
			// 能被 URL 正常解析则直接返回，端口细节交给构造函数处理
			new URL(trimmedUrl)
			return trimmedUrl
		} catch {
			// 解析失败，兜底按 hostname 处理
			return this.parseHostname(trimmedUrl)
		}
	}

	/**
	 * 处理 hostname 形式输入并补全协议。
	 *
	 * @param hostname 原始主机名（可能含端口）
	 * @returns        规范化 URL（默认补 `http://`）
	 */
	private parseHostname(hostname: string): string {
		if (hostname.includes(":")) {
			// 携带端口时仅补协议
			return hostname.startsWith("http") ? hostname : `http://${hostname}`
		} else {
			// 不含端口时补协议，端口后续由协议默认值决定
			return `http://${hostname}`
		}
	}

	/**
	 * 获取集合信息。
	 *
	 * @returns 集合信息；不存在或读取失败时返回 null
	 */
	private async getCollectionInfo(): Promise<Schemas["CollectionInfo"] | null> {
		try {
			const collectionInfo = await this.client.getCollection(this.collectionName)
			return collectionInfo
		} catch (error: unknown) {
			if (error instanceof Error) {
				console.warn(
					`[QdrantVectorStore] Warning during getCollectionInfo for "${this.collectionName}". Collection may not exist or another error occurred:`,
					error.message,
				)
			}
			return null
		}
	}

	/**
	 * 初始化向量存储集合。
	 *
	 * 逻辑：
	 * 1) 检查集合是否存在
	 * 2) 不存在则创建
	 * 3) 存在则校验维度，不一致时重建
	 * 4) 创建 payload 索引
	 *
	 * @returns `true` 表示创建了新集合，`false` 表示复用了已有集合
	 */
	async initialize(): Promise<boolean> {
		let created = false
		try {
			const collectionInfo = await this.getCollectionInfo()

			if (collectionInfo === null) {
				// 未获取到集合信息（通常是不存在），直接创建
				await this.client.createCollection(this.collectionName, {
					vectors: {
						size: this.vectorSize,
						distance: this.DISTANCE_METRIC,
						on_disk: true,
					},
					hnsw_config: {
						m: 64,
						ef_construct: 512,
						on_disk: true,
					},
				})
				created = true
			} else {
				// 集合已存在，校验向量维度是否与当前模型一致
				const vectorsConfig = collectionInfo.config?.params?.vectors
				let existingVectorSize: number

				if (typeof vectorsConfig === "number") {
					existingVectorSize = vectorsConfig
				} else if (
					vectorsConfig &&
					typeof vectorsConfig === "object" &&
					"size" in vectorsConfig &&
					typeof vectorsConfig.size === "number"
				) {
					existingVectorSize = vectorsConfig.size
				} else {
					existingVectorSize = 0 // 未知配置结构时使用兜底值
				}

				if (existingVectorSize === this.vectorSize) {
					created = false // 维度一致，可复用
				} else {
					// 维度不一致：必须重建集合，避免写入/检索维度错误
					created = await this._recreateCollectionWithNewDimension(existingVectorSize)
				}
			}

			// 为检索过滤字段创建 payload 索引，提高查询性能
			await this._createPayloadIndexes()
			return created
		} catch (error: any) {
			const errorMessage = error?.message || error
			console.error(
				`[QdrantVectorStore] Failed to initialize Qdrant collection "${this.collectionName}":`,
				errorMessage,
			)

			// 若已是“维度不匹配重建失败”的包装错误，直接透传
			if (error instanceof Error && error.cause !== undefined) {
				throw error
			}

			// 其余错误包装为更友好的连接失败提示
			throw new Error(
				t("embeddings:vectorStore.qdrantConnectionFailed", { qdrantUrl: this.qdrantUrl, errorMessage }),
			)
		}
	}

	/**
	 * 在维度变化时重建集合，并尽可能提供可诊断的分阶段错误信息。
	 *
	 * @param existingVectorSize 现有集合维度
	 * @returns                  `true` 表示成功按新维度创建
	 */
	private async _recreateCollectionWithNewDimension(existingVectorSize: number): Promise<boolean> {
		console.warn(
			`[QdrantVectorStore] Collection ${this.collectionName} exists with vector size ${existingVectorSize}, but expected ${this.vectorSize}. Recreating collection.`,
		)

		let deletionSucceeded = false
		let recreationAttempted = false

		try {
			// 1) 删除旧集合
			console.log(`[QdrantVectorStore] Deleting existing collection ${this.collectionName}...`)
			await this.client.deleteCollection(this.collectionName)
			deletionSucceeded = true
			console.log(`[QdrantVectorStore] Successfully deleted collection ${this.collectionName}`)

			// 2) 短暂等待，确保删除操作已落地
			await new Promise((resolve) => setTimeout(resolve, 100))

			// 3) 二次确认集合确实已删除
			const verificationInfo = await this.getCollectionInfo()
			if (verificationInfo !== null) {
				throw new Error("Collection still exists after deletion attempt")
			}

			// 4) 按新维度重建集合
			console.log(
				`[QdrantVectorStore] Creating new collection ${this.collectionName} with vector size ${this.vectorSize}...`,
			)
			recreationAttempted = true
			await this.client.createCollection(this.collectionName, {
				vectors: {
					size: this.vectorSize,
					distance: this.DISTANCE_METRIC,
					on_disk: true,
				},
				hnsw_config: {
					m: 64,
					ef_construct: 512,
					on_disk: true,
				},
			})
			console.log(`[QdrantVectorStore] Successfully created new collection ${this.collectionName}`)
			return true
		} catch (recreationError) {
			const errorMessage = recreationError instanceof Error ? recreationError.message : String(recreationError)

			// 按失败阶段拼装上下文，帮助快速定位是“删失败”“校验失败”还是“重建失败”
			let contextualErrorMessage: string
			if (!deletionSucceeded) {
				contextualErrorMessage = `Failed to delete existing collection with vector size ${existingVectorSize}. ${errorMessage}`
			} else if (!recreationAttempted) {
				contextualErrorMessage = `Deleted existing collection but failed verification step. ${errorMessage}`
			} else {
				contextualErrorMessage = `Deleted existing collection but failed to create new collection with vector size ${this.vectorSize}. ${errorMessage}`
			}

			console.error(
				`[QdrantVectorStore] CRITICAL: Failed to recreate collection ${this.collectionName} for dimension change (${existingVectorSize} -> ${this.vectorSize}). ${contextualErrorMessage}`,
			)

			// 对外抛出更可读的错误信息
			const dimensionMismatchError = new Error(
				t("embeddings:vectorStore.vectorDimensionMismatch", {
					errorMessage: contextualErrorMessage,
				}),
			)

			// 保留底层错误作为 cause
			dimensionMismatchError.cause = recreationError
			throw dimensionMismatchError
		}
	}

	/**
	 * 为集合创建 payload 索引（容错处理）。
	 * 已存在索引不视为错误，其他错误仅告警不阻塞主流程。
	 */
	private async _createPayloadIndexes(): Promise<void> {
		// 为 type 字段建索引，用于排除 metadata 点
		try {
			await this.client.createPayloadIndex(this.collectionName, {
				field_name: "type",
				field_schema: "keyword",
			})
		} catch (indexError: any) {
			const errorMessage = (indexError?.message || "").toLowerCase()
			if (!errorMessage.includes("already exists")) {
				console.warn(
					`[QdrantVectorStore] Could not create payload index for type on ${this.collectionName}. Details:`,
					indexError?.message || indexError,
				)
			}
		}

		// 为 pathSegments.0~4 建索引，加速目录前缀过滤
		for (let i = 0; i <= 4; i++) {
			try {
				await this.client.createPayloadIndex(this.collectionName, {
					field_name: `pathSegments.${i}`,
					field_schema: "keyword",
				})
			} catch (indexError: any) {
				const errorMessage = (indexError?.message || "").toLowerCase()
				if (!errorMessage.includes("already exists")) {
					console.warn(
						`[QdrantVectorStore] Could not create payload index for pathSegments.${i} on ${this.collectionName}. Details:`,
						indexError?.message || indexError,
					)
				}
			}
		}
	}

	/**
	 * 批量写入（upsert）向量点。
	 *
	 * 会将 filePath 预处理为 `pathSegments`，以便后续按目录/文件快速过滤。
	 *
	 * @param points 待写入的点集合
	 */
	async upsertPoints(
		points: Array<{
			id: string
			vector: number[]
			payload: Record<string, any>
		}>,
	): Promise<void> {
		try {
			// 在写入前补充 pathSegments（按路径段拆分后的可过滤字段）
			const processedPoints = points.map((point) => {
				if (point.payload?.filePath) {
					const segments = point.payload.filePath.split(path.sep).filter(Boolean)
					const pathSegments = segments.reduce(
						(acc: Record<string, string>, segment: string, index: number) => {
							acc[index.toString()] = segment
							return acc
						},
						{},
					)
					return {
						...point,
						payload: {
							...point.payload,
							pathSegments,
						},
					}
				}
				return point
			})

			await this.client.upsert(this.collectionName, {
				points: processedPoints,
				wait: true,
			})
		} catch (error) {
			console.error("Failed to upsert points:", error)
			throw error
		}
	}

	/**
	 * 校验 payload 是否具备最小可用字段。
	 *
	 * @param payload 待校验 payload
	 * @returns       是否为合法检索结果 payload
	 */
	private isPayloadValid(payload: Record<string, unknown> | null | undefined): payload is Payload {
		if (!payload) {
			return false
		}
		const validKeys = ["filePath", "codeChunk", "startLine", "endLine"]
		const hasValidKeys = validKeys.every((key) => key in payload)
		return hasValidKeys
	}

	/**
	 * 执行相似向量检索。
	 *
	 * @param queryVector     查询向量
	 * @param directoryPrefix 可选目录前缀（用于路径过滤）
	 * @param minScore        可选最小相似度阈值
	 * @param maxResults      可选最大返回数量
	 * @returns               检索结果列表
	 */
	async search(
		queryVector: number[],
		directoryPrefix?: string,
		minScore?: number,
		maxResults?: number,
	): Promise<VectorStoreSearchResult[]> {
		try {
			let filter:
				| {
						must: Array<{ key: string; match: { value: string } }>
						must_not?: Array<{ key: string; match: { value: string } }>
				  }
				| undefined = undefined

			if (directoryPrefix) {
				// 统一路径分隔符并标准化，便于跨平台匹配
				const normalizedPrefix = path.posix.normalize(directoryPrefix.replace(/\\/g, "/"))
				// "." / "./" 代表当前目录，不做过滤（全工作区检索）
				if (normalizedPrefix === "." || normalizedPrefix === "./") {
					filter = undefined
				} else {
					// 去掉前导 "./" 并按路径段拆分，映射为 pathSegments.N 过滤条件
					const cleanedPrefix = path.posix.normalize(
						normalizedPrefix.startsWith("./") ? normalizedPrefix.slice(2) : normalizedPrefix,
					)
					const segments = cleanedPrefix.split("/").filter(Boolean)
					if (segments.length > 0) {
						filter = {
							must: segments.map((segment, index) => ({
								key: `pathSegments.${index}`,
								match: { value: segment },
							})),
						}
					}
				}
			}

			// 查询时始终排除 metadata 点，避免占用 top-k 名额
			const metadataExclusion = {
				must_not: [{ key: "type", match: { value: "metadata" } }],
			}

			const mergedFilter = filter
				? { ...filter, must_not: [...(filter.must_not || []), ...metadataExclusion.must_not] }
				: metadataExclusion

			const searchRequest = {
				query: queryVector,
				filter: mergedFilter,
				score_threshold: minScore ?? DEFAULT_SEARCH_MIN_SCORE,
				limit: maxResults ?? DEFAULT_MAX_SEARCH_RESULTS,
				params: {
					hnsw_ef: 128,
					exact: false,
				},
				with_payload: {
					include: ["filePath", "codeChunk", "startLine", "endLine", "pathSegments"],
				},
			}

			// 执行查询并过滤掉 payload 不完整的点
			const operationResult = await this.client.query(this.collectionName, searchRequest)
			const filteredPoints = operationResult.points.filter((p) => this.isPayloadValid(p.payload))

			return filteredPoints as VectorStoreSearchResult[]
		} catch (error) {
			console.error("Failed to search points:", error)
			throw error
		}
	}

	/**
	 * 按单个文件路径删除向量点。
	 *
	 * @param filePath 文件路径
	 */
	async deletePointsByFilePath(filePath: string): Promise<void> {
		return this.deletePointsByMultipleFilePaths([filePath])
	}

	/**
	 * 按多个文件路径删除向量点。
	 * 内部将每个文件路径转换为 pathSegments 过滤条件执行批量删除。
	 *
	 * @param filePaths 文件路径数组
	 */
	async deletePointsByMultipleFilePaths(filePaths: string[]): Promise<void> {
		if (filePaths.length === 0) {
			return
		}

		try {
			// 先检查集合是否存在，避免无意义删除请求
			const collectionExists = await this.collectionExists()
			if (!collectionExists) {
				console.warn(
					`[QdrantVectorStore] Skipping deletion - collection "${this.collectionName}" does not exist`,
				)
				return
			}

			const workspaceRoot = this.workspacePath

			// 基于 pathSegments 构建删除过滤条件，与 upsert 时写入结构保持一致
			const filters = filePaths.map((filePath) => {
				// 重要：upsert 存的是相对路径，这里必须统一转为相对路径再匹配
				const relativePath = path.isAbsolute(filePath) ? path.relative(workspaceRoot, filePath) : filePath

				// 规范化路径，避免分隔符差异
				const normalizedRelativePath = path.normalize(relativePath)

				// 按路径段拆分，保持与 upsert 的 pathSegments 生成逻辑一致
				const segments = normalizedRelativePath.split(path.sep).filter(Boolean)

				// 仅当所有路径段都命中时才删除，避免误删同名前缀文件
				const mustConditions = segments.map((segment, index) => ({
					key: `pathSegments.${index}`,
					match: { value: segment },
				}))

				return { must: mustConditions }
			})

			// 多文件时使用 should（OR）组合
			const filter = filters.length === 1 ? filters[0] : { should: filters }

			await this.client.delete(this.collectionName, {
				filter,
				wait: true,
			})
		} catch (error: any) {
			// 记录更完整的错误上下文，便于定位删除失败原因
			const errorMessage = error?.message || String(error)
			const errorStatus = error?.status || error?.response?.status || error?.statusCode
			const errorDetails = error?.response?.data || error?.data || ""

			console.error(`[QdrantVectorStore] Failed to delete points by file paths:`, {
				error: errorMessage,
				status: errorStatus,
				details: errorDetails,
				collection: this.collectionName,
				fileCount: filePaths.length,
				// 仅记录少量样本路径，避免日志过大
				samplePaths: filePaths.slice(0, 3),
			})
		}
	}

	/**
	 * 删除整个集合。
	 */
	async deleteCollection(): Promise<void> {
		try {
			// 先判存再删，避免无意义异常
			if (await this.collectionExists()) {
				await this.client.deleteCollection(this.collectionName)
			}
		} catch (error) {
			console.error(`[QdrantVectorStore] Failed to delete collection ${this.collectionName}:`, error)
			throw error // 继续上抛给调用方处理
		}
	}

	/**
	 * 清空集合中的所有点（保留集合结构）。
	 */
	async clearCollection(): Promise<void> {
		try {
			await this.client.delete(this.collectionName, {
				filter: {
					must: [],
				},
				wait: true,
			})
		} catch (error) {
			console.error("Failed to clear collection:", error)
			throw error
		}
	}

	/**
	 * 检查集合是否存在。
	 *
	 * @returns 是否存在
	 */
	async collectionExists(): Promise<boolean> {
		const collectionInfo = await this.getCollectionInfo()
		return collectionInfo !== null
	}

	/**
	 * 检查集合是否有可用索引数据。
	 *
	 * 判定顺序：
	 * 1) 集合存在且 points_count > 0
	 * 2) 若存在 metadata 标记点，则以 `indexing_complete` 为准
	 * 3) 若无标记点（兼容旧版本），退化为 points_count > 0
	 *
	 * @returns 是否有可用索引数据
	 */
	async hasIndexedData(): Promise<boolean> {
		try {
			const collectionInfo = await this.getCollectionInfo()
			if (!collectionInfo) {
				return false
			}
			// 集合点数为 0 直接返回 false
			const pointsCount = collectionInfo.points_count ?? 0
			if (pointsCount === 0) {
				return false
			}

			// 读取固定 metadata 点，判断本次索引是否完整结束
			const metadataId = uuidv5("__indexing_metadata__", QDRANT_CODE_BLOCK_NAMESPACE)
			const metadataPoints = await this.client.retrieve(this.collectionName, {
				ids: [metadataId],
			})

			// 有标记点时，以显式完成标记为准
			if (metadataPoints.length > 0) {
				return metadataPoints[0].payload?.indexing_complete === true
			}

			// 向后兼容：旧索引无 metadata 标记，回退到 points_count 逻辑
			console.log(
				"[QdrantVectorStore] No indexing metadata marker found. Using backward compatibility mode (checking points_count > 0).",
			)
			return pointsCount > 0
		} catch (error) {
			console.warn("[QdrantVectorStore] Failed to check if collection has data:", error)
			return false
		}
	}

	/**
	 * 将索引状态标记为“已完成”。
	 * 通常在全量扫描或增量扫描成功结束后调用。
	 */
	async markIndexingComplete(): Promise<void> {
		try {
			// 使用固定 UUID 写 metadata 点，保证每次写入覆盖同一条记录
			const metadataId = uuidv5("__indexing_metadata__", QDRANT_CODE_BLOCK_NAMESPACE)

			await this.client.upsert(this.collectionName, {
				points: [
					{
						id: metadataId,
						vector: new Array(this.vectorSize).fill(0),
						payload: {
							type: "metadata",
							indexing_complete: true,
							completed_at: Date.now(),
						},
					},
				],
				wait: true,
			})
			console.log("[QdrantVectorStore] Marked indexing as complete")
		} catch (error) {
			console.error("[QdrantVectorStore] Failed to mark indexing as complete:", error)
			throw error
		}
	}

	/**
	 * 将索引状态标记为“进行中/未完成”。
	 * 通常在索引开始前调用。
	 */
	async markIndexingIncomplete(): Promise<void> {
		try {
			// 使用固定 UUID 写 metadata 点，保证状态单点更新
			const metadataId = uuidv5("__indexing_metadata__", QDRANT_CODE_BLOCK_NAMESPACE)

			await this.client.upsert(this.collectionName, {
				points: [
					{
						id: metadataId,
						vector: new Array(this.vectorSize).fill(0),
						payload: {
							type: "metadata",
							indexing_complete: false,
							started_at: Date.now(),
						},
					},
				],
				wait: true,
			})
			console.log("[QdrantVectorStore] Marked indexing as incomplete (in progress)")
		} catch (error) {
			console.error("[QdrantVectorStore] Failed to mark indexing as incomplete:", error)
			throw error
		}
	}
}
