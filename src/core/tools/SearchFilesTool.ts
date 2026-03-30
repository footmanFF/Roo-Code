import path from "path"

import { type ClineSayTool } from "@roo-code/types"

import { Task } from "../task/Task"
import { getReadablePath } from "../../utils/path"
import { isPathOutsideWorkspace } from "../../utils/pathUtils"
import { regexSearchFiles } from "../../services/ripgrep"
import type { ToolUse } from "../../shared/tools"

import { BaseTool, ToolCallbacks } from "./BaseTool"

// search_files 工具的入参结构
interface SearchFilesParams {
	path: string             // 要搜索的目录路径（相对路径）
	regex: string            // 正则表达式（Rust regex 语法）
	file_pattern?: string | null  // 可选的 glob 文件类型过滤，如 "*.ts"
}

/**
 * search_files 工具：使用 ripgrep 在指定目录内执行正则检索，
 * 返回匹配的文件路径、行号及前后各 1 行上下文。
 */
export class SearchFilesTool extends BaseTool<"search_files"> {
	readonly name = "search_files" as const

	async execute(params: SearchFilesParams, task: Task, callbacks: ToolCallbacks): Promise<void> {
		const { askApproval, handleError, pushToolResult } = callbacks

		const relDirPath = params.path
		const regex = params.regex
		const filePattern = params.file_pattern || undefined

		// 必填参数校验：缺少 path 时记录错误并提前返回
		if (!relDirPath) {
			task.consecutiveMistakeCount++
			task.recordToolError("search_files")
			task.didToolFailInCurrentTurn = true
			pushToolResult(await task.sayAndCreateMissingParamError("search_files", "path"))
			return
		}

		// 必填参数校验：缺少 regex 时记录错误并提前返回
		if (!regex) {
			task.consecutiveMistakeCount++
			task.recordToolError("search_files")
			task.didToolFailInCurrentTurn = true
			pushToolResult(await task.sayAndCreateMissingParamError("search_files", "regex"))
			return
		}

		// 参数合法，重置连续错误计数
		task.consecutiveMistakeCount = 0

		// 将相对路径解析为绝对路径，并判断是否超出工作区范围
		const absolutePath = path.resolve(task.cwd, relDirPath)
		const isOutsideWorkspace = isPathOutsideWorkspace(absolutePath)

		// 构造用于展示给用户的消息体（显示在 UI 审批弹窗中）
		const sharedMessageProps: ClineSayTool = {
			tool: "searchFiles",
			path: getReadablePath(task.cwd, relDirPath),
			regex: regex,
			filePattern: filePattern,
			isOutsideWorkspace,
		}

		try {
			// 调用 ripgrep 执行实际的正则检索
			const results = await regexSearchFiles(task.cwd, absolutePath, regex, filePattern, task.rooIgnoreController)

			// 将检索结果附加到消息体，请求用户审批
			const completeMessage = JSON.stringify({ ...sharedMessageProps, content: results } satisfies ClineSayTool)
			const didApprove = await askApproval("tool", completeMessage)

			// 用户拒绝则不返回结果
			if (!didApprove) {
				return
			}

			// 将检索结果写入 tool_result，供 LLM 下一轮使用
			pushToolResult(results)
		} catch (error) {
			await handleError("searching files", error as Error)
		}
	}

	/**
	 * 流式响应期间的局部渲染处理：
	 * LLM 还在输出参数时，实时向 UI 推送占位消息（内容为空），
	 * 让用户看到工具正在被调用，而无需等到参数完全解析完毕。
	 */
	override async handlePartial(task: Task, block: ToolUse<"search_files">): Promise<void> {
		const relDirPath = block.params.path
		const regex = block.params.regex
		const filePattern = block.params.file_pattern

		const absolutePath = relDirPath ? path.resolve(task.cwd, relDirPath) : task.cwd
		const isOutsideWorkspace = isPathOutsideWorkspace(absolutePath)

		const sharedMessageProps: ClineSayTool = {
			tool: "searchFiles",
			path: getReadablePath(task.cwd, relDirPath ?? ""),
			regex: regex ?? "",
			filePattern: filePattern ?? "",
			isOutsideWorkspace,
		}

		// content 为空字符串，表示结果尚未就绪，仅用于 UI 实时展示
		const partialMessage = JSON.stringify({ ...sharedMessageProps, content: "" } satisfies ClineSayTool)
		await task.ask("tool", partialMessage, block.partial).catch(() => {})
	}
}

// 单例导出，供工具注册表使用
export const searchFilesTool = new SearchFilesTool()
