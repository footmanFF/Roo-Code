import * as path from "path"
import { VectorStoreSearchResult } from "./interfaces"
import { IEmbedder } from "./interfaces/embedder"
import { IVectorStore } from "./interfaces/vector-store"
import { CodeIndexConfigManager } from "./config-manager"
import { CodeIndexStateManager } from "./state-manager"
import { TelemetryService } from "@roo-code/telemetry"
import { TelemetryEventName } from "@roo-code/types"

/**
 * 代码索引检索服务。
 * 负责将查询文本转为向量，并在向量库中执行相似度检索。
 */
export class CodeIndexSearchService {
	/**
	 * @param configManager 配置管理器（提供检索阈值、结果数量上限、功能开关等）
	 * @param stateManager  状态管理器（读取当前索引状态，记录错误状态）
	 * @param embedder      向量化组件（将查询文本转为 embedding）
	 * @param vectorStore   向量存储组件（执行相似度检索）
	 */
	constructor(
		private readonly configManager: CodeIndexConfigManager,
		private readonly stateManager: CodeIndexStateManager,
		private readonly embedder: IEmbedder,
		private readonly vectorStore: IVectorStore,
	) {}

	/**
	 * 在代码索引中执行语义检索。
	 *
	 * 执行步骤：
	 * 1) 校验功能开关与配置状态
	 * 2) 校验索引系统状态（仅 Indexed / Indexing 允许检索）
	 * 3) 将 query 向量化
	 * 4) 可选目录前缀标准化
	 * 5) 调用向量库检索并返回结果
	 *
	 * @param query           检索查询文本
	 * @param directoryPrefix 可选目录前缀，用于限制检索范围
	 * @returns               向量检索结果数组
	 * @throws Error          功能未启用、索引不可用或检索过程失败时抛出
	 */
	public async searchIndex(query: string, directoryPrefix?: string): Promise<VectorStoreSearchResult[]> {
		// 功能开关或基础配置不满足时直接失败（如缺失 key / 向量库地址）
		if (!this.configManager.isFeatureEnabled || !this.configManager.isFeatureConfigured) {
			throw new Error("Code index feature is disabled or not configured.")
		}

		// 从配置读取检索参数：最小分数阈值 + 最大返回条数
		const minScore = this.configManager.currentSearchMinScore
		const maxResults = this.configManager.currentSearchMaxResults

		const currentState = this.stateManager.getCurrentStatus().systemStatus
		if (currentState !== "Indexed" && currentState !== "Indexing") {
			// 允许在 Indexing 过程中检索；其余状态视为未就绪
			throw new Error(`Code index is not ready for search. Current state: ${currentState}`)
		}

		try {
			// 先将自然语言查询转为向量，后续用于向量相似度检索
			const embeddingResponse = await this.embedder.createEmbeddings([query])
			const vector = embeddingResponse?.embeddings[0]
			if (!vector) {
				throw new Error("Failed to generate embedding for query.")
			}

			// 目录前缀标准化，避免跨平台路径分隔符差异影响过滤命中
			let normalizedPrefix: string | undefined = undefined
			if (directoryPrefix) {
				normalizedPrefix = path.normalize(directoryPrefix)
			}

			// 在向量库执行检索：向量 + 目录前缀过滤 + 分数阈值 + 结果条数上限
			const results = await this.vectorStore.search(vector, normalizedPrefix, minScore, maxResults)
			return results
		} catch (error) {
			// 检索失败时：记录日志 + 更新系统状态为 Error + 上报遥测，再继续向上抛错
			console.error("[CodeIndexSearchService] Error during search:", error)
			this.stateManager.setSystemState("Error", `Search failed: ${(error as Error).message}`)

			// 上报错误遥测，便于线上问题诊断
			TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
				error: (error as Error).message,
				stack: (error as Error).stack,
				location: "searchIndex",
			})

			// 更新状态后继续抛错，由上层决定提示文案与恢复策略
			throw error
		}
	}
}
