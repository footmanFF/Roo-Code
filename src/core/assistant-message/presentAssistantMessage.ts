import { serializeError } from "serialize-error"
import { Anthropic } from "@anthropic-ai/sdk"

import type { ToolName, ClineAsk, ToolProgressStatus } from "@roo-code/types"
import { ConsecutiveMistakeError, TelemetryEventName } from "@roo-code/types"
import { TelemetryService } from "@roo-code/telemetry"
import { customToolRegistry } from "@roo-code/core"

import { t } from "../../i18n"

import { defaultModeSlug, getModeBySlug } from "../../shared/modes"
import type { ToolParamName, ToolResponse, ToolUse, McpToolUse } from "../../shared/tools"

import { AskIgnoredError } from "../task/AskIgnoredError"
import { Task } from "../task/Task"

import { listFilesTool } from "../tools/ListFilesTool"
import { readFileTool } from "../tools/ReadFileTool"
import { readCommandOutputTool } from "../tools/ReadCommandOutputTool"
import { writeToFileTool } from "../tools/WriteToFileTool"
import { editTool } from "../tools/EditTool"
import { searchReplaceTool } from "../tools/SearchReplaceTool"
import { editFileTool } from "../tools/EditFileTool"
import { applyPatchTool } from "../tools/ApplyPatchTool"
import { searchFilesTool } from "../tools/SearchFilesTool"
import { executeCommandTool } from "../tools/ExecuteCommandTool"
import { useMcpToolTool } from "../tools/UseMcpToolTool"
import { accessMcpResourceTool } from "../tools/accessMcpResourceTool"
import { askFollowupQuestionTool } from "../tools/AskFollowupQuestionTool"
import { switchModeTool } from "../tools/SwitchModeTool"
import { attemptCompletionTool, AttemptCompletionCallbacks } from "../tools/AttemptCompletionTool"
import { newTaskTool } from "../tools/NewTaskTool"
import { updateTodoListTool } from "../tools/UpdateTodoListTool"
import { runSlashCommandTool } from "../tools/RunSlashCommandTool"
import { skillTool } from "../tools/SkillTool"
import { generateImageTool } from "../tools/GenerateImageTool"
import { applyDiffTool as applyDiffToolClass } from "../tools/ApplyDiffTool"
import { isValidToolName, validateToolUse } from "../tools/validateToolUse"
import { codebaseSearchTool } from "../tools/CodebaseSearchTool"

import { formatResponse } from "../prompts/responses"
import { sanitizeToolUseId } from "../../utils/tool-id"

/**
 * Processes and presents assistant message content to the user interface.
 *
 * This function is the core message handling system that:
 * - Sequentially processes content blocks from the assistant's response.
 * - Displays text content to the user.
 * - Executes tool use requests with appropriate user approval.
 * - Manages the flow of conversation by determining when to proceed to the next content block.
 * - Coordinates file system checkpointing for modified files.
 * - Controls the conversation state to determine when to continue to the next request.
 *
 * The function uses a locking mechanism to prevent concurrent execution and handles
 * partial content blocks during streaming. It's designed to work with the streaming
 * API response pattern, where content arrives incrementally and needs to be processed
 * as it becomes available.
 */

/**
 * 处理并呈现 LLM 返回的单条 assistant 消息内容块（content block）。
 *
 * 本函数是 Agent 循环的核心分发器，负责：
 * 1. 将 LLM 流式返回的文本（text）实时渲染到 UI
 * 2. 将 LLM 的工具调用（tool_use / mcp_tool_use）分发到对应的工具实现执行
 * 3. 收集工具执行结果，写入 `cline.userMessageContent`，供下一轮 LLM 请求使用
 *
 * 本函数由流式读取循环（attemptApiRequest）在每个新 content block 到达时触发，
 * 也可能在工具执行完成后递归调用自身以处理下一个 block。
 *
 * @param cline 当前 Task 实例，包含会话状态、消息历史、流控标志等所有运行时上下文
 * @returns Promise<void>（无返回值；工具结果通过 cline.userMessageContent 传递）
 */
export async function presentAssistantMessage(cline: Task) {
	// 任务已中止时直接抛错，终止整个调用链
	if (cline.abort) {
		throw new Error(`[Task#presentAssistantMessage] task ${cline.taskId}.${cline.instanceId} aborted`)
	}

	// 防重入锁：本函数可能由流式事件和工具回调并发触发，
	// 若已有一次调用正在执行，则标记有待处理更新并提前返回，
	// 待当前调用结束时（函数末尾）检查标志后再次调用自身。
	if (cline.presentAssistantMessageLocked) {
		cline.presentAssistantMessageHasPendingUpdates = true
		return
	}

	cline.presentAssistantMessageLocked = true
	cline.presentAssistantMessageHasPendingUpdates = false

	// 越界检查：索引超出 assistantMessageContent 范围，说明当前轮次的所有 block 已处理完毕。
	// 若流也已读完，则将 userMessageContentReady 置 true，
	// 唤醒主循环中 pWaitFor(userMessageContentReady) 的等待，触发下一轮 LLM 调用。
	if (cline.currentStreamingContentIndex >= cline.assistantMessageContent.length) {
		if (cline.didCompleteReadingStream) {
			cline.userMessageContentReady = true
		}

		cline.presentAssistantMessageLocked = false
		return
	}

	let block: any
	try {
		// 性能优化：使用浅拷贝而非深克隆。
		// 整个函数对 block 只读，不修改其属性，浅拷贝足以防止流式更新期间引用被替换。
		// 相比深克隆可减少 80-90% 的开销（每次调用节省 5-100ms）。
		block = { ...cline.assistantMessageContent[cline.currentStreamingContentIndex] }
	} catch (error) {
		console.error(`ERROR cloning block:`, error)
		console.error(
			`Block content:`,
			JSON.stringify(cline.assistantMessageContent[cline.currentStreamingContentIndex], null, 2),
		)
		cline.presentAssistantMessageLocked = false
		return
	}

	// 按 content block 类型分发处理逻辑
	switch (block.type) {
		case "mcp_tool_use":
			await handleMcpToolUseCase(cline, block as McpToolUse)
			break
		case "text":
			await handleTextCase(cline, block)
			break
		case "tool_use":
			await handleToolUseCase(cline, block)
			break
	}

	// 在 switch/case 执行完毕后立即释放锁，
	// 此时下一个 block 可能已在 assistantMessageContent 中就位，
	// 若持锁再调用 presentAssistantMessage 会导致死锁。
	cline.presentAssistantMessageLocked = false

	// block 完整执行完毕（或用户拒绝/本轮已用工具导致中止）时，推进 block 索引并决定后续动作。
	// 注意：工具被拒绝时流被中断，主循环 pWaitFor(userMessageContentReady) 会一直等待；
	// 后续对 presentAssistantMessage 的调用会因 didRejectTool=true 跳过执行，
	// 直到 contentIndex 到达末尾并将 userMessageContentReady 置为 true，主循环才继续。
	if (!block.partial || cline.didRejectTool || cline.didAlreadyUseTool) {
		if (cline.currentStreamingContentIndex === cline.assistantMessageContent.length - 1) {
			// 当前 block 是本轮最后一个，且已执行完毕，唤醒主循环
			cline.userMessageContentReady = true
		}

		// 无论是否越界都先自增，下次流式事件触发时会从新索引位置读取
		cline.currentStreamingContentIndex++

		if (cline.currentStreamingContentIndex < cline.assistantMessageContent.length) {
			// 还有更多 block 等待处理，递归调用处理下一个
			presentAssistantMessage(cline)
			return
		} else {
			// 已越界：若流也已结束，则标记 userMessageContentReady，
			// 处理 assistantMessageContent 为空或提前清空的边界情况
			if (cline.didCompleteReadingStream) {
				cline.userMessageContentReady = true
			}
		}
	}

	// 当前 block 仍为流式中间态，但锁释放期间已有新数据到达（标志位被置 true），
	// 立即再次触发处理，避免遗漏更新
	if (cline.presentAssistantMessageHasPendingUpdates) {
		presentAssistantMessage(cline)
	}
}

/**
 * 处理 `mcp_tool_use` 类型的内容块。
 *
 * 核心职责：
 * - 将原生 MCP tool block 适配为 `use_mcp_tool` 统一执行路径
 * - 处理用户审批、异常兜底、tool_result 回填与去重
 * - 保持 Native 协议下 tool_use/tool_result 的配对完整性
 */
async function handleMcpToolUseCase(cline: Task, mcpBlock: McpToolUse): Promise<void> {
	if (cline.didRejectTool) {
		// 用户已拒绝过前一个工具，本工具跳过执行。
		// Native 协议要求每个 tool_use 都必须有对应的 tool_result，否则 API 会报错，
		// 因此即使跳过执行也要推送一条错误类型的 tool_result。
		const toolCallId = mcpBlock.id
		const errorMessage = !mcpBlock.partial
			? `Skipping MCP tool ${mcpBlock.name} due to user rejecting a previous tool.`
			: `MCP tool ${mcpBlock.name} was interrupted and not executed due to user rejecting a previous tool.`

		if (toolCallId) {
			cline.pushToolResultToUserContent({
				type: "tool_result",
				tool_use_id: sanitizeToolUseId(toolCallId),
				content: errorMessage,
				is_error: true,
			})
		}
		return
	}

	// 防重复推送标志：同一个 tool_use_id 只能有一条 tool_result
	let hasToolResult = false
	const toolCallId = mcpBlock.id

	// 缓存用户审批时附带的反馈文字/图片，待工具执行完成后一并合并到 tool_result 中
	let approvalFeedback: { text: string; images?: string[] } | undefined

	const pushToolResult = (content: ToolResponse) => {
		if (hasToolResult) {
			console.warn(`[presentAssistantMessage] Skipping duplicate tool_result for mcp_tool_use: ${toolCallId}`)
			return
		}

		let resultContent: string
		let imageBlocks: Anthropic.ImageBlockParam[] = []

		// 将 content 统一转为纯文本 + 图片块两部分
		if (typeof content === "string") {
			resultContent = content || "(tool did not return anything)"
		} else {
			const textBlocks = content.filter((item) => item.type === "text")
			imageBlocks = content.filter((item) => item.type === "image") as Anthropic.ImageBlockParam[]
			resultContent =
				textBlocks.map((item) => (item as Anthropic.TextBlockParam).text).join("\n") ||
				"(tool did not return anything)"
		}

		// 将审批时的用户反馈文字/图片合并到工具结果的最前面（GitHub #10465）
		if (approvalFeedback) {
			const feedbackText = formatResponse.toolApprovedWithFeedback(approvalFeedback.text)
			resultContent = `${feedbackText}\n\n${resultContent}`

			if (approvalFeedback.images) {
				const feedbackImageBlocks = formatResponse.imageBlocks(approvalFeedback.images)
				imageBlocks = [...feedbackImageBlocks, ...imageBlocks]
			}
		}

		if (toolCallId) {
			// 文本结果写入 tool_result block
			cline.pushToolResultToUserContent({
				type: "tool_result",
				tool_use_id: sanitizeToolUseId(toolCallId),
				content: resultContent,
			})

			// 图片单独追加（API 不支持在 tool_result.content 中直接放图片块）
			if (imageBlocks.length > 0) {
				cline.userMessageContent.push(...imageBlocks)
			}
		}

		hasToolResult = true
	}

	const askApproval = async (
		type: ClineAsk,
		partialMessage?: string,
		progressStatus?: ToolProgressStatus,
		isProtected?: boolean,
	) => {
		const { response, text, images } = await cline.ask(
			type,
			partialMessage,
			false,
			progressStatus,
			isProtected || false,
		)

		if (response !== "yesButtonClicked") {
			if (text) {
				await cline.say("user_feedback", text, images)
				pushToolResult(formatResponse.toolResult(formatResponse.toolDeniedWithFeedback(text), images))
			} else {
				pushToolResult(formatResponse.toolDenied())
			}
			cline.didRejectTool = true
			return false
		}

		// 用户点了允许但附带了文字，暂存到 approvalFeedback，
		// 不在此处推送 tool_result（避免重复），等工具执行完成后由 pushToolResult 合并
		if (text) {
			await cline.say("user_feedback", text, images)
			approvalFeedback = { text, images }
		}

		return true
	}

	const handleError = async (action: string, error: Error) => {
		if (error instanceof AskIgnoredError) {
			return
		}
		const errorString = `Error ${action}: ${JSON.stringify(serializeError(error))}`
		await cline.say("error", `Error ${action}:\n${error.message ?? JSON.stringify(serializeError(error), null, 2)}`)
		pushToolResult(formatResponse.toolError(errorString))
	}

	// 完整 block（非流式中间态）才记录工具使用统计，避免重复计数
	if (!mcpBlock.partial) {
		cline.recordToolUsage("use_mcp_tool")
		TelemetryService.instance.captureToolUsage(cline.taskId, "use_mcp_tool")
	}

	// 将解析出的经过 sanitize 处理的 serverName 还原为 MCP Hub 中注册的原始名称
	// （例如：解析结果为 "my_server"，实际注册名为 "my server"）
	const mcpHub = cline.providerRef.deref()?.getMcpHub()
	let resolvedServerName = mcpBlock.serverName
	if (mcpHub) {
		const originalName = mcpHub.findServerNameBySanitizedName(mcpBlock.serverName)
		if (originalName) {
			resolvedServerName = originalName
		}
	}

	// 将 mcp_tool_use block 转换为 use_mcp_tool 能处理的 ToolUse 格式，复用同一套执行逻辑
	const syntheticToolUse: ToolUse<"use_mcp_tool"> = {
		type: "tool_use",
		id: mcpBlock.id,
		name: "use_mcp_tool",
		params: {
			server_name: resolvedServerName,
			tool_name: mcpBlock.toolName,
			arguments: JSON.stringify(mcpBlock.arguments),
		},
		partial: mcpBlock.partial,
		nativeArgs: {
			server_name: resolvedServerName,
			tool_name: mcpBlock.toolName,
			arguments: mcpBlock.arguments,
		},
	}

	await useMcpToolTool.handle(cline, syntheticToolUse, {
		askApproval,
		handleError,
		pushToolResult,
	})
}

/**
 * 处理 `text` 类型内容块：
 * - 清理 `<thinking>` 标签
 * - 在流式与非流式场景下统一推送到 UI
 */
async function handleTextCase(cline: Task, block: any): Promise<void> {
	// 用户已拒绝工具或本轮已执行过工具时，跳过后续文本 block，
	// 防止将工具中断后 LLM 续写的文字展示给用户
	if (cline.didRejectTool || cline.didAlreadyUseTool) {
		return
	}

	let content = block.content

	if (content) {
		// 剥除 LLM 可能流式输出的 <thinking>...</thinking> 标签，
		// Markdown 渲染器会自动删除该标签内容，需在展示前主动清理
		content = content.replace(/<thinking>\s?/g, "")
		content = content.replace(/\s?<\/thinking>/g, "")
	}

	// 将文本内容实时推送到 UI（block.partial=true 时为流式增量更新）
	await cline.say("text", content, undefined, block.partial)
}

/**
 * 处理 `tool_use` 类型内容块。
 *
 * 核心职责：
 * - 校验工具调用合法性与参数完整性
 * - 统一封装审批、错误处理、tool_result 生成
 * - 分发到具体工具实现并记录统计/遥测
 */
async function handleToolUseCase(cline: Task, block: any): Promise<void> {
	// 项目只支持 Native Tool Calling 协议（OpenAI/Anthropic Function Calling），
	// 不再支持 XML 标签形式的工具调用。tool_use block 必须携带唯一 id。
	const toolCallId = (block as any).id as string | undefined
	if (!toolCallId) {
		const errorMessage =
			"Invalid tool call: missing tool_use.id. XML tool calls are no longer supported. Remove any XML tool markup (e.g. <read_file>...</read_file>) and use native tool calling instead."
		try {
			if (typeof (cline as any).recordToolError === "function" && typeof (block as any).name === "string") {
				;(cline as any).recordToolError((block as any).name as ToolName, errorMessage)
			}
		} catch {
			// 尽力记录，失败不影响主流程
		}
		cline.consecutiveMistakeCount++
		await cline.say("error", errorMessage)
		cline.userMessageContent.push({ type: "text", text: errorMessage })
		cline.didAlreadyUseTool = true
		return
	}

	const state = await cline.providerRef.deref()?.getState()
	const { mode, customModes, experiments: stateExperiments, disabledTools } = state ?? {}

	const toolDescription = (): string => {
		switch (block.name) {
			case "execute_command":
				return `[${block.name} for '${block.params.command}']`
			case "read_file":
				if (block.nativeArgs) {
					return readFileTool.getReadFileToolDescription(block.name, block.nativeArgs)
				}
				return readFileTool.getReadFileToolDescription(block.name, block.params)
			case "write_to_file":
				return `[${block.name} for '${block.params.path}']`
			case "apply_diff":
				return block.params?.path ? `[${block.name} for '${block.params.path}']` : `[${block.name}]`
			case "search_files":
				return `[${block.name} for '${block.params.regex}'${block.params.file_pattern ? ` in '${block.params.file_pattern}'` : ""}]`
			case "edit":
			case "search_and_replace":
			case "search_replace":
			case "edit_file":
				return `[${block.name} for '${block.params.file_path}']`
			case "apply_patch":
				return `[${block.name}]`
			case "list_files":
				return `[${block.name} for '${block.params.path}']`
			case "use_mcp_tool":
			case "access_mcp_resource":
				return `[${block.name} for '${block.params.server_name}']`
			case "ask_followup_question":
				return `[${block.name} for '${block.params.question}']`
			case "attempt_completion":
			case "update_todo_list":
				return `[${block.name}]`
			case "switch_mode":
				return `[${block.name} to '${block.params.mode_slug}'${block.params.reason ? ` because: ${block.params.reason}` : ""}]`
			case "codebase_search":
				return `[${block.name} for '${block.params.query}']`
			case "read_command_output":
				return `[${block.name} for '${block.params.artifact_id}']`
			case "new_task": {
				const mode = block.params.mode ?? defaultModeSlug
				const message = block.params.message ?? "(no message)"
				const modeName = getModeBySlug(mode, customModes)?.name ?? mode
				return `[${block.name} in ${modeName} mode: '${message}']`
			}
			case "run_slash_command":
				return `[${block.name} for '${block.params.command}'${block.params.args ? ` with args: ${block.params.args}` : ""}]`
			case "skill":
				return `[${block.name} for '${block.params.skill}'${block.params.args ? ` with args: ${block.params.args}` : ""}]`
			case "generate_image":
				return `[${block.name} for '${block.params.path}']`
			default:
				return `[${block.name}]`
		}
	}

	if (cline.didRejectTool) {
		const errorMessage = !block.partial
			? `Skipping tool ${toolDescription()} due to user rejecting a previous tool.`
			: `Tool ${toolDescription()} was interrupted and not executed due to user rejecting a previous tool.`

		cline.pushToolResultToUserContent({
			type: "tool_result",
			tool_use_id: sanitizeToolUseId(toolCallId),
			content: errorMessage,
			is_error: true,
		})
		return
	}

	let hasToolResult = false

	if (!block.partial) {
		const customTool = stateExperiments?.customTools ? customToolRegistry.get(block.name) : undefined
		const isKnownTool = isValidToolName(String(block.name), stateExperiments)
		if (isKnownTool && !block.nativeArgs && !customTool) {
			const errorMessage =
				`Invalid tool call for '${block.name}': missing nativeArgs. ` +
				`This usually means the model streamed invalid or incomplete arguments and the call could not be finalized.`

			cline.consecutiveMistakeCount++
			try {
				cline.recordToolError(block.name as ToolName, errorMessage)
			} catch {
				// 尽力记录
			}

			cline.pushToolResultToUserContent({
				type: "tool_result",
				tool_use_id: sanitizeToolUseId(toolCallId),
				content: formatResponse.toolError(errorMessage),
				is_error: true,
			})
			return
		}
	}

	let approvalFeedback: { text: string; images?: string[] } | undefined

	const pushToolResult = (content: ToolResponse) => {
		if (hasToolResult) {
			console.warn(`[presentAssistantMessage] Skipping duplicate tool_result for tool_use_id: ${toolCallId}`)
			return
		}

		let resultContent: string
		let imageBlocks: Anthropic.ImageBlockParam[] = []
		if (typeof content === "string") {
			resultContent = content || "(tool did not return anything)"
		} else {
			const textBlocks = content.filter((item) => item.type === "text")
			imageBlocks = content.filter((item) => item.type === "image") as Anthropic.ImageBlockParam[]
			resultContent =
				textBlocks.map((item) => (item as Anthropic.TextBlockParam).text).join("\n") ||
				"(tool did not return anything)"
		}

		if (approvalFeedback) {
			const feedbackText = formatResponse.toolApprovedWithFeedback(approvalFeedback.text)
			resultContent = `${feedbackText}\n\n${resultContent}`
			if (approvalFeedback.images) {
				const feedbackImageBlocks = formatResponse.imageBlocks(approvalFeedback.images)
				imageBlocks = [...feedbackImageBlocks, ...imageBlocks]
			}
		}

		cline.pushToolResultToUserContent({
			type: "tool_result",
			tool_use_id: sanitizeToolUseId(toolCallId),
			content: resultContent,
		})

		if (imageBlocks.length > 0) {
			cline.userMessageContent.push(...imageBlocks)
		}

		hasToolResult = true
	}

	const askApproval = async (
		type: ClineAsk,
		partialMessage?: string,
		progressStatus?: ToolProgressStatus,
		isProtected?: boolean,
	) => {
		const { response, text, images } = await cline.ask(
			type,
			partialMessage,
			false,
			progressStatus,
			isProtected || false,
		)

		if (response !== "yesButtonClicked") {
			if (text) {
				await cline.say("user_feedback", text, images)
				pushToolResult(formatResponse.toolResult(formatResponse.toolDeniedWithFeedback(text), images))
			} else {
				pushToolResult(formatResponse.toolDenied())
			}
			cline.didRejectTool = true
			return false
		}

		if (text) {
			await cline.say("user_feedback", text, images)
			approvalFeedback = { text, images }
		}

		return true
	}

	const askFinishSubTaskApproval = async () => {
		const toolMessage = JSON.stringify({ tool: "finishTask" })
		return await askApproval("tool", toolMessage)
	}

	const handleError = async (action: string, error: Error) => {
		if (error instanceof AskIgnoredError) {
			return
		}
		const errorString = `Error ${action}: ${JSON.stringify(serializeError(error))}`
		await cline.say("error", `Error ${action}:\n${error.message ?? JSON.stringify(serializeError(error), null, 2)}`)
		pushToolResult(formatResponse.toolError(errorString))
	}

	if (!block.partial) {
		const isCustomTool = stateExperiments?.customTools && customToolRegistry.has(block.name)
		const recordName = isCustomTool ? "custom_tool" : block.name
		cline.recordToolUsage(recordName)
		TelemetryService.instance.captureToolUsage(cline.taskId, recordName)

		if (block.name === "read_file" && block.usedLegacyFormat) {
			const modelInfo = cline.api.getModel()
			TelemetryService.instance.captureEvent(TelemetryEventName.READ_FILE_LEGACY_FORMAT_USED, {
				taskId: cline.taskId,
				model: modelInfo?.id,
			})
		}
	}

	if (!block.partial) {
		const modelInfo = cline.api.getModel()
		const rawIncludedTools = modelInfo?.info?.includedTools
		const { resolveToolAlias } = await import("../prompts/tools/filter-tools-for-mode")
		const includedTools = rawIncludedTools?.map((tool) => resolveToolAlias(tool))

		try {
			const toolRequirements =
				disabledTools?.reduce(
					(acc: Record<string, boolean>, tool: string) => {
						acc[tool] = false
						const resolvedToolName = resolveToolAlias(tool)
						acc[resolvedToolName] = false
						return acc
					},
					{} as Record<string, boolean>,
				) ?? {}

			validateToolUse(
				block.name as ToolName,
				mode ?? defaultModeSlug,
				customModes ?? [],
				toolRequirements,
				block.params,
				stateExperiments,
				includedTools,
			)
		} catch (error: any) {
			cline.consecutiveMistakeCount++
			const errorContent = formatResponse.toolError(error.message)
			cline.pushToolResultToUserContent({
				type: "tool_result",
				tool_use_id: sanitizeToolUseId(toolCallId),
				content: typeof errorContent === "string" ? errorContent : "(validation error)",
				is_error: true,
			})
			return
		}
	}

	if (!block.partial) {
		const repetitionCheck = cline.toolRepetitionDetector.check(block)

		if (!repetitionCheck.allowExecution && repetitionCheck.askUser) {
			const { response, text, images } = await cline.ask(
				repetitionCheck.askUser.messageKey as ClineAsk,
				repetitionCheck.askUser.messageDetail.replace("{toolName}", block.name),
			)

			if (response === "messageResponse") {
				cline.userMessageContent.push(
					{
						type: "text" as const,
						text: `Tool repetition limit reached. User feedback: ${text}`,
					},
					...formatResponse.imageBlocks(images),
				)

				await cline.say("user_feedback", text, images)
			}

			TelemetryService.instance.captureConsecutiveMistakeError(cline.taskId)
			TelemetryService.instance.captureException(
				new ConsecutiveMistakeError(
					`Tool repetition limit reached for ${block.name}`,
					cline.taskId,
					cline.consecutiveMistakeCount,
					cline.consecutiveMistakeLimit,
					"tool_repetition",
					cline.apiConfiguration.apiProvider,
					cline.api.getModel().id,
				),
			)

			pushToolResult(
				formatResponse.toolError(
					`Tool call repetition limit reached for ${block.name}. Please try a different approach.`,
				),
			)
			return
		}
	}

	switch (block.name) {
		case "write_to_file":
			await checkpointSaveAndMark(cline)
			await writeToFileTool.handle(cline, block as ToolUse<"write_to_file">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "update_todo_list":
			await updateTodoListTool.handle(cline, block as ToolUse<"update_todo_list">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "apply_diff":
			await checkpointSaveAndMark(cline)
			await applyDiffToolClass.handle(cline, block as ToolUse<"apply_diff">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "edit":
		case "search_and_replace":
			await checkpointSaveAndMark(cline)
			await editTool.handle(cline, block as ToolUse<"edit">, { askApproval, handleError, pushToolResult })
			break
		case "search_replace":
			await checkpointSaveAndMark(cline)
			await searchReplaceTool.handle(cline, block as ToolUse<"search_replace">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "edit_file":
			await checkpointSaveAndMark(cline)
			await editFileTool.handle(cline, block as ToolUse<"edit_file">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "apply_patch":
			await checkpointSaveAndMark(cline)
			await applyPatchTool.handle(cline, block as ToolUse<"apply_patch">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "read_file":
			await readFileTool.handle(cline, block as ToolUse<"read_file">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "list_files":
			await listFilesTool.handle(cline, block as ToolUse<"list_files">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "codebase_search":
			await codebaseSearchTool.handle(cline, block as ToolUse<"codebase_search">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "search_files":
			await searchFilesTool.handle(cline, block as ToolUse<"search_files">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "execute_command":
			await executeCommandTool.handle(cline, block as ToolUse<"execute_command">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "read_command_output":
			await readCommandOutputTool.handle(cline, block as ToolUse<"read_command_output">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "use_mcp_tool":
			await useMcpToolTool.handle(cline, block as ToolUse<"use_mcp_tool">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "access_mcp_resource":
			await accessMcpResourceTool.handle(cline, block as ToolUse<"access_mcp_resource">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "ask_followup_question":
			await askFollowupQuestionTool.handle(cline, block as ToolUse<"ask_followup_question">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "switch_mode":
			await switchModeTool.handle(cline, block as ToolUse<"switch_mode">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "new_task":
			await checkpointSaveAndMark(cline)
			await newTaskTool.handle(cline, block as ToolUse<"new_task">, {
				askApproval,
				handleError,
				pushToolResult,
				toolCallId: block.id,
			})
			break
		case "attempt_completion": {
			const completionCallbacks: AttemptCompletionCallbacks = {
				askApproval,
				handleError,
				pushToolResult,
				askFinishSubTaskApproval,
				toolDescription,
			}
			await attemptCompletionTool.handle(cline, block as ToolUse<"attempt_completion">, completionCallbacks)
			break
		}
		case "run_slash_command":
			await runSlashCommandTool.handle(cline, block as ToolUse<"run_slash_command">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		case "skill":
			await skillTool.handle(cline, block as ToolUse<"skill">, { askApproval, handleError, pushToolResult })
			break
		case "generate_image":
			await checkpointSaveAndMark(cline)
			await generateImageTool.handle(cline, block as ToolUse<"generate_image">, {
				askApproval,
				handleError,
				pushToolResult,
			})
			break
		default: {
			// 未知工具名或自定义工具的处理入口。
			// Native 协议要求每个 tool_use 都必须有 tool_result，此处也不例外。
			if (block.partial) {
				break
			}

			const customTool = stateExperiments?.customTools ? customToolRegistry.get(block.name) : undefined
			if (customTool) {
				try {
					let customToolArgs
					if (customTool.parameters) {
						try {
							customToolArgs = customTool.parameters.parse(block.nativeArgs || block.params || {})
						} catch (parseParamsError: any) {
							const message = `Custom tool "${block.name}" argument validation failed: ${parseParamsError.message}`
							console.error(message)
							cline.consecutiveMistakeCount++
							await cline.say("error", message)
							pushToolResult(formatResponse.toolError(message))
							break
						}
					}

					const result = await customTool.execute(customToolArgs, {
						mode: mode ?? defaultModeSlug,
						task: cline,
					})

					console.log(
						`${customTool.name}.execute(): ${JSON.stringify(customToolArgs)} -> ${JSON.stringify(result)}`,
					)
					pushToolResult(result)
					cline.consecutiveMistakeCount = 0
				} catch (executionError: any) {
					cline.consecutiveMistakeCount++
					cline.recordToolError("custom_tool", executionError.message)
					await handleError(`executing custom tool "${block.name}"`, executionError)
				}

				break
			}

			const errorMessage = `Unknown tool "${block.name}". This tool does not exist. Please use one of the available tools.`
			cline.consecutiveMistakeCount++
			cline.recordToolError(block.name as ToolName, errorMessage)
			await cline.say("error", t("tools:unknownToolError", { toolName: block.name }))
			cline.pushToolResultToUserContent({
				type: "tool_result",
				tool_use_id: sanitizeToolUseId(toolCallId),
				content: formatResponse.toolError(errorMessage),
				is_error: true,
			})
			break
		}
	}
}

/**
 * save checkpoint and mark done in the current streaming task.
 * @param task The Task instance to checkpoint save and mark.
 * @returns
 */
async function checkpointSaveAndMark(task: Task) {
	if (task.currentStreamingDidCheckpoint) {
		return
	}
	try {
		await task.checkpointSave(true)
		task.currentStreamingDidCheckpoint = true
	} catch (error) {
		console.error(`[Task#presentAssistantMessage] Error saving checkpoint: ${error.message}`, error)
	}
}
