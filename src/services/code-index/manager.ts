import * as vscode from "vscode"
import { ContextProxy } from "../../core/config/ContextProxy"
import { VectorStoreSearchResult } from "./interfaces"
import { IndexingState } from "./interfaces/manager"
import { CodeIndexConfigManager } from "./config-manager"
import { CodeIndexStateManager } from "./state-manager"
import { CodeIndexServiceFactory } from "./service-factory"
import { CodeIndexSearchService } from "./search-service"
import { CodeIndexOrchestrator } from "./orchestrator"
import { CacheManager } from "./cache-manager"
import { RooIgnoreController } from "../../core/ignore/RooIgnoreController"
import fs from "fs/promises"
import ignore from "ignore"
import path from "path"
import { t } from "../../i18n"
import { TelemetryService } from "@roo-code/telemetry"
import { TelemetryEventName } from "@roo-code/types"

export class CodeIndexManager {
	// --- Singleton Implementation ---
	private static instances = new Map<string, CodeIndexManager>() // Map workspace path to instance

	// Specialized class instances
	private _configManager: CodeIndexConfigManager | undefined
	private readonly _stateManager: CodeIndexStateManager
	private _serviceFactory: CodeIndexServiceFactory | undefined
	private _orchestrator: CodeIndexOrchestrator | undefined
	private _searchService: CodeIndexSearchService | undefined
	private _cacheManager: CacheManager | undefined

	// Flag to prevent race conditions during error recovery
	private _isRecoveringFromError = false

	/**
	 * 获取指定工作区的 CodeIndexManager 单例实例。
	 *
	 * 规则：
	 * - 传入 `workspacePath` 时，按该路径绑定实例。
	 * - 未传入时，优先取当前激活编辑器所在工作区；否则取第一个工作区。
	 * - 若实例不存在则创建并缓存，后续复用同一实例。
	 *
	 * @param context       VS Code 扩展上下文
	 * @param workspacePath 可选工作区路径
	 * @returns             对应工作区的管理器实例；无工作区时返回 undefined
	 */
	public static getInstance(context: vscode.ExtensionContext, workspacePath?: string): CodeIndexManager | undefined {
		// Resolve the workspace folder to get both fsPath and the real URI
		let folder: vscode.WorkspaceFolder | undefined

		if (workspacePath) {
			// 显式路径模式：尝试在已打开工作区中找到匹配项
			folder = vscode.workspace.workspaceFolders?.find((f) => f.uri.fsPath === workspacePath)
		} else {
			// 自动推断模式：优先当前编辑器所属工作区
			const activeEditor = vscode.window.activeTextEditor
			if (activeEditor) {
				folder = vscode.workspace.getWorkspaceFolder(activeEditor.document.uri)
			}
			if (!folder) {
				// 再兜底到第一个工作区
				const workspaceFolders = vscode.workspace.workspaceFolders
				if (!workspaceFolders || workspaceFolders.length === 0) {
					return undefined
				}
				folder = workspaceFolders[0]
			}
			workspacePath = folder.uri.fsPath
		}

		if (!CodeIndexManager.instances.has(workspacePath)) {
			// folder may be undefined when workspacePath was provided but doesn't match
			// any workspace folder (e.g. cwd passed from a tool). Fall back to file:// URI.
			const folderUri =
				folder?.uri ??
				({
					fsPath: workspacePath,
					scheme: "file",
					authority: "",
					path: workspacePath,
					toString: () => `file://${workspacePath}`,
				} as unknown as vscode.Uri)
			CodeIndexManager.instances.set(workspacePath, new CodeIndexManager(workspacePath, folderUri, context))
		}
		return CodeIndexManager.instances.get(workspacePath)!
	}

	/** 返回当前已创建的所有工作区实例。 */
	public static getAllInstances(): CodeIndexManager[] {
		return Array.from(CodeIndexManager.instances.values())
	}

	/** 释放并清空所有工作区实例。 */
	public static disposeAll(): void {
		for (const instance of CodeIndexManager.instances.values()) {
			instance.dispose()
		}
		CodeIndexManager.instances.clear()
	}

	private readonly workspacePath: string
	private readonly _folderUri: vscode.Uri
	private readonly context: vscode.ExtensionContext

	// Private constructor for singleton pattern
	private constructor(workspacePath: string, folderUri: vscode.Uri, context: vscode.ExtensionContext) {
		this.workspacePath = workspacePath
		this._folderUri = folderUri
		this.context = context
		this._stateManager = new CodeIndexStateManager()
	}

	// --- Public API ---

	/**
	 * Returns the workspaceState key for per-folder indexing enablement,
	 * keyed by the real workspace folder URI so local/remote schemes cannot collide.
	 */
	private _workspaceEnabledKey(): string {
		return "codeIndexWorkspaceEnabled:" + this._folderUri.toString(true)
	}

	/** 当前工作区是否启用代码索引（先读工作区显式配置，否则回退到全局默认值）。 */
	public get isWorkspaceEnabled(): boolean {
		const explicit = this.context.workspaceState.get<boolean | undefined>(this._workspaceEnabledKey(), undefined)
		if (explicit !== undefined) return explicit
		return this.autoEnableDefault
	}

	/** 设置当前工作区的代码索引启用状态。 */
	public async setWorkspaceEnabled(enabled: boolean): Promise<void> {
		await this.context.workspaceState.update(this._workspaceEnabledKey(), enabled)
	}

	/** 新工作区自动启用索引的全局默认值。 */
	public get autoEnableDefault(): boolean {
		return this.context.globalState.get("codeIndexAutoEnableDefault", true)
	}

	/** 设置新工作区自动启用索引的全局默认值。 */
	public async setAutoEnableDefault(enabled: boolean): Promise<void> {
		await this.context.globalState.update("codeIndexAutoEnableDefault", enabled)
	}

	/** 订阅索引进度更新事件。 */
	public get onProgressUpdate() {
		return this._stateManager.onProgressUpdate
	}

	/** 断言核心依赖已初始化，未初始化则抛错。 */
	private assertInitialized() {
		if (!this._configManager || !this._orchestrator || !this._searchService || !this._cacheManager) {
			throw new Error("CodeIndexManager not initialized. Call initialize() first.")
		}
	}

	/** 当前索引系统状态（未启用时固定为 Standby）。 */
	public get state(): IndexingState {
		if (!this.isFeatureEnabled) {
			return "Standby"
		}
		this.assertInitialized()
		return this._orchestrator!.state
	}

	/** 功能开关是否启用（来自配置）。 */
	public get isFeatureEnabled(): boolean {
		return this._configManager?.isFeatureEnabled ?? false
	}

	/** 功能是否完成最小配置（如 API Key、向量库地址）。 */
	public get isFeatureConfigured(): boolean {
		return this._configManager?.isFeatureConfigured ?? false
	}

	/** 是否已完成初始化（可安全执行索引/检索）。 */
	public get isInitialized(): boolean {
		try {
			this.assertInitialized()
			return true
		} catch (error) {
			return false
		}
	}

	/**
	 * 初始化管理器及其依赖服务。
	 * 必须在调用其他核心能力（索引/检索）前执行。
	 *
	 * 关键步骤：
	 * 1) 加载配置并判断是否需重启服务
	 * 2) 检查功能开关、工作区可用性、工作区级启用状态
	 * 3) 初始化缓存并按需重建服务
	 * 4) 按条件触发索引启动/重启
	 *
	 * @returns `{ requiresRestart }`：表示配置变更是否要求重建服务
	 */
	public async initialize(contextProxy: ContextProxy): Promise<{ requiresRestart: boolean }> {
		// 1. ConfigManager Initialization and Configuration Loading
		if (!this._configManager) {
			this._configManager = new CodeIndexConfigManager(contextProxy)
		}
		// Load configuration once to get current state and restart requirements
		const { requiresRestart } = await this._configManager.loadConfiguration()

		// 2. Check if feature is enabled
		if (!this.isFeatureEnabled) {
			// 功能关闭时只确保 watcher 停止，不创建昂贵服务
			if (this._orchestrator) {
				this._orchestrator.stopWatcher()
			}
			return { requiresRestart }
		}

		// 3. Check if workspace is available
		const workspacePath = this.workspacePath
		if (!workspacePath) {
			this._stateManager.setSystemState("Standby", "No workspace folder open")
			return { requiresRestart }
		}

		// 4. Check workspace-level enablement (before creating expensive services)
		if (!this.isWorkspaceEnabled) {
			// 工作区级禁用：进入待机状态，不继续初始化底层服务
			this._stateManager.setSystemState("Standby", "Indexing not enabled for this workspace")
			return { requiresRestart }
		}

		// 5. CacheManager Initialization
		if (!this._cacheManager) {
			this._cacheManager = new CacheManager(this.context, this.workspacePath)
			await this._cacheManager.initialize()
		}

		// 6. Determine if Core Services Need Recreation
		const needsServiceRecreation = !this._serviceFactory || requiresRestart

		if (needsServiceRecreation) {
			// 配置变更或首次启动时，重建整套服务依赖
			await this._recreateServices()
		}

		// 7. Handle Indexing Start/Restart
		const shouldStartOrRestartIndexing =
			requiresRestart ||
			(needsServiceRecreation && (!this._orchestrator || this._orchestrator.state !== "Indexing"))

		if (shouldStartOrRestartIndexing) {
			// 后台启动（或重启）索引流程
			this._orchestrator?.startIndexing()
		}

		return { requiresRestart }
	}

	/**
	 * 启动索引流程（首次扫描 + 文件监听）。
	 * 若当前处于 Error 状态，会先执行恢复流程。
	 *
	 * @important 该方法会触发长生命周期后台任务，不应在 UI 主流程中阻塞等待其完成。
	 */
	public async startIndexing(): Promise<void> {
		if (!this.isFeatureEnabled || !this.isWorkspaceEnabled) {
			return
		}

		// Check if we're in error state and recover if needed
		const currentStatus = this.getCurrentStatus()
		if (currentStatus.systemStatus === "Error") {
			await this.recoverFromError()

			// 恢复会清空服务实例，需由上层再次 initialize()
			return
		}

		this.assertInitialized()
		await this._orchestrator!.startIndexing()
	}

	/**
	 * 停止索引任务与文件监听。
	 */
	public stopIndexing(): void {
		if (this._orchestrator) {
			this._orchestrator.stopIndexing()
		}
	}

	/**
	 * 仅停止文件监听（不清理索引数据）。
	 */
	public stopWatcher(): void {
		if (!this.isFeatureEnabled) {
			return
		}
		if (this._orchestrator) {
			this._orchestrator.stopWatcher()
		}
	}

	/**
	 * 从错误态恢复：清空错误状态并重置内部服务引用。
	 * 该方法不会自动重新初始化，调用方应在恢复后重新执行 initialize()。
	 *
	 * 设计目标：在网络异常、配置异常等可恢复错误后，确保下次初始化从“干净状态”开始。
	 *
	 * @remarks
	 * - 幂等：非错误态调用也安全
	 * - 不自动重启索引
	 * - 通过并发保护避免重复恢复引发竞态
	 */
	public async recoverFromError(): Promise<void> {
		// Prevent race conditions from multiple rapid recovery attempts
		if (this._isRecoveringFromError) {
			return
		}

		this._isRecoveringFromError = true
		try {
			// Clear error state
			this._stateManager.setSystemState("Standby", "")
		} catch (error) {
			// Log error but continue with recovery - clearing service instances is more important
			console.error("Failed to clear error state during recovery:", error)
		} finally {
			// 无论清错是否成功，都强制清空服务引用，保证下次 initialize 全量重建
			this._configManager = undefined
			this._serviceFactory = undefined
			this._orchestrator = undefined
			this._searchService = undefined

			// Reset the flag after recovery is complete
			this._isRecoveringFromError = false
		}
	}

	/**
	 * 释放当前实例资源。
	 */
	public dispose(): void {
		this.stopIndexing()
		this._stateManager.dispose()
	}

	/**
	 * 清空索引数据：
	 * - 停止索引/监听
	 * - 清理向量库集合
	 * - 清理本地缓存文件
	 */
	public async clearIndexData(): Promise<void> {
		if (!this.isFeatureEnabled) {
			return
		}
		this.assertInitialized()
		await this._orchestrator!.clearIndexData()
		await this._cacheManager!.clearCacheFile()
	}

	// --- Private Helpers ---

	/** 获取当前状态快照（附加工作区维度信息）。 */
	public getCurrentStatus() {
		const status = this._stateManager.getCurrentStatus()
		return {
			...status,
			workspacePath: this.workspacePath,
			workspaceEnabled: this.isWorkspaceEnabled,
			autoEnableDefault: this.autoEnableDefault,
		}
	}

	/**
	 * 执行语义检索。
	 *
	 * @param query           检索查询文本
	 * @param directoryPrefix 可选目录前缀过滤
	 * @returns               向量检索结果列表；功能关闭时返回空数组
	 */
	public async searchIndex(query: string, directoryPrefix?: string): Promise<VectorStoreSearchResult[]> {
		if (!this.isFeatureEnabled) {
			return []
		}
		this.assertInitialized()
		return this._searchService!.searchIndex(query, directoryPrefix)
	}

	/**
	 * 按当前配置重建核心服务（工厂、扫描器、向量库、编排器、检索服务）。
	 * 同时用于初始化阶段和设置变更后的重建阶段。
	 */
	private async _recreateServices(): Promise<void> {
		// Stop watcher if it exists
		if (this._orchestrator) {
			this.stopWatcher()
		}
		// Clear existing services to ensure clean state
		this._orchestrator = undefined
		this._searchService = undefined

		// (Re)Initialize service factory
		this._serviceFactory = new CodeIndexServiceFactory(
			this._configManager!,
			this.workspacePath,
			this._cacheManager!,
		)

		const ignoreInstance = ignore()
		const workspacePath = this.workspacePath

		if (!workspacePath) {
			this._stateManager.setSystemState("Standby", "")
			return
		}

		// Create .gitignore instance
		const ignorePath = path.join(workspacePath, ".gitignore")
		try {
			const content = await fs.readFile(ignorePath, "utf8")
			ignoreInstance.add(content)
			// 避免 .gitignore 文件自身被纳入索引
			ignoreInstance.add(".gitignore")
		} catch (error) {
			// Should never happen: reading file failed even though it exists
			console.error("Unexpected error loading .gitignore:", error)
			TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
				error: error instanceof Error ? error.message : String(error),
				stack: error instanceof Error ? error.stack : undefined,
				location: "_recreateServices",
			})
		}

		// Create RooIgnoreController instance
		const rooIgnoreController = new RooIgnoreController(workspacePath)
		await rooIgnoreController.initialize()

		// (Re)Create shared service instances
		const { embedder, vectorStore, scanner, fileWatcher } = this._serviceFactory.createServices(
			this.context,
			this._cacheManager!,
			ignoreInstance,
			rooIgnoreController,
		)

		// Validate embedder configuration before proceeding
		const validationResult = await this._serviceFactory.validateEmbedder(embedder)
		if (!validationResult.valid) {
			// Embedding 侧配置无效时直接切换 Error，阻断后续索引流程
			const errorMessage = validationResult.error || "Embedder configuration validation failed"
			this._stateManager.setSystemState("Error", errorMessage)
			throw new Error(errorMessage)
		}

		// (Re)Initialize orchestrator
		this._orchestrator = new CodeIndexOrchestrator(
			this._configManager!,
			this._stateManager,
			this.workspacePath,
			this._cacheManager!,
			vectorStore,
			scanner,
			fileWatcher,
		)

		// (Re)Initialize search service
		this._searchService = new CodeIndexSearchService(
			this._configManager!,
			this._stateManager,
			embedder,
			vectorStore,
		)

		// Clear any error state after successful recreation
		this._stateManager.setSystemState("Standby", "")
	}

	/**
	 * 处理代码索引相关设置变更。
	 * 当配置要求重启服务时，执行服务重建以应用新配置。
	 */
	public async handleSettingsChange(): Promise<void> {
		if (this._configManager) {
			const { requiresRestart } = await this._configManager.loadConfiguration()

			const isFeatureEnabled = this.isFeatureEnabled
			const isFeatureConfigured = this.isFeatureConfigured

			// If feature is disabled, stop the service (including any active scan)
			if (!isFeatureEnabled) {
				// 配置切换为关闭后，立即进入待机态并停止后台任务
				this.stopIndexing()
				this._stateManager.setSystemState("Standby", "Code indexing is disabled")
				return
			}

			if (requiresRestart && isFeatureEnabled && isFeatureConfigured) {
				try {
					// Ensure cacheManager is initialized before recreating services
					if (!this._cacheManager) {
						this._cacheManager = new CacheManager(this.context, this.workspacePath)
						await this._cacheManager.initialize()
					}

					// Recreate services with new configuration
					await this._recreateServices()
				} catch (error) {
					// _recreateServices 内已设置 Error 态，这里负责记录遥测并向上抛出
					console.error("Failed to recreate services:", error)
					TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
						error: error instanceof Error ? error.message : String(error),
						stack: error instanceof Error ? error.stack : undefined,
						location: "handleSettingsChange",
					})
					// Re-throw the error so the caller knows validation failed
					throw error
				}
			}
		}
	}
}
