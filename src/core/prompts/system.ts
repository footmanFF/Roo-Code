import * as vscode from "vscode"

import { type ModeConfig, type PromptComponent, type CustomModePrompts, type TodoItem } from "@roo-code/types"

import { Mode, modes, defaultModeSlug, getModeBySlug, getGroupName, getModeSelection } from "../../shared/modes"
import { DiffStrategy } from "../../shared/tools"
import { formatLanguage } from "../../shared/language"
import { isEmpty } from "../../utils/object"

import { McpHub } from "../../services/mcp/McpHub"
import { CodeIndexManager } from "../../services/code-index/manager"
import { SkillsManager } from "../../services/skills/SkillsManager"

import type { SystemPromptSettings } from "./types"
import {
	getRulesSection,
	getSystemInfoSection,
	getObjectiveSection,
	getSharedToolUseSection,
	getToolUseGuidelinesSection,
	getCapabilitiesSection,
	getModesSection,
	addCustomInstructions,
	markdownFormattingSection,
	getSkillsSection,
} from "./sections"

/**
 * 获取指定模式的自定义 Prompt 片段，并过滤空配置。
 *
 * @param customModePrompts 自定义模式 Prompt 配置映射
 * @param mode              当前模式 slug
 * @returns                 有效 PromptComponent；若为空对象则返回 undefined
 */
export function getPromptComponent(
	customModePrompts: CustomModePrompts | undefined,
	mode: string,
): PromptComponent | undefined {
	const component = customModePrompts?.[mode]
	// 空对象视为“未配置”，统一返回 undefined，避免覆盖默认行为
	if (isEmpty(component)) {
		return undefined
	}
	return component
}

/**
 * 组装 system prompt 的核心实现。
 *
 * 该函数将“角色定义 + 工具使用规范 + 能力说明 + 模式说明 + 规则 + 系统信息 + 目标 + 自定义指令”
 * 拼接为最终系统提示词文本。
 *
 * @param context                  VS Code 扩展上下文
 * @param cwd                      当前工作目录
 * @param supportsComputerUse      是否支持 computer use（保留参数，兼容调用链）
 * @param mode                     当前模式 slug
 * @param mcpHub                   MCP Hub（用于决定是否注入 MCP 能力说明）
 * @param diffStrategy             diff 策略（保留参数，兼容调用链）
 * @param promptComponent          模式级自定义 Prompt 片段
 * @param customModeConfigs        自定义模式配置
 * @param globalCustomInstructions 全局自定义指令
 * @param experiments              实验特性开关
 * @param language                 语言设置
 * @param rooIgnoreInstructions    rooignore 指令
 * @param settings                 system prompt 相关设置
 * @param todoList                 待办列表（保留参数，兼容调用链）
 * @param modelId                  当前模型 ID（保留参数，兼容调用链）
 * @param skillsManager            技能管理器（用于生成可用技能段落）
 * @returns                        拼接后的 system prompt 文本
 */
async function generatePrompt(
	context: vscode.ExtensionContext,
	cwd: string,
	supportsComputerUse: boolean,
	mode: Mode,
	mcpHub?: McpHub,
	diffStrategy?: DiffStrategy,
	promptComponent?: PromptComponent,
	customModeConfigs?: ModeConfig[],
	globalCustomInstructions?: string,
	experiments?: Record<string, boolean>,
	language?: string,
	rooIgnoreInstructions?: string,
	settings?: SystemPromptSettings,
	todoList?: TodoItem[],
	modelId?: string,
	skillsManager?: SkillsManager,
): Promise<string> {
	if (!context) {
		throw new Error("Extension context is required for generating system prompt")
	}

	// 解析完整模式配置：优先自定义模式，其次内置模式，最后兜底第一个模式
	const modeConfig = getModeBySlug(mode, customModeConfigs) || modes.find((m) => m.slug === mode) || modes[0]
	// 获取角色定义与基础指令（会被后续 addCustomInstructions 进一步加工）
	const { roleDefinition, baseInstructions } = getModeSelection(mode, promptComponent, customModeConfigs)

	// 仅当“当前模式包含 mcp 工具组”且“实际存在 MCP 服务”时，才注入 MCP 能力说明
	const hasMcpGroup = modeConfig.groups.some((groupEntry) => getGroupName(groupEntry) === "mcp")
	const hasMcpServers = mcpHub && mcpHub.getServers().length > 0
	const shouldIncludeMcp = hasMcpGroup && hasMcpServers

	// 预留：用于与代码索引能力保持一致的上下文入口（当前文件内不直接使用）
	const codeIndexManager = CodeIndexManager.getInstance(context, cwd)

	// 工具调用协议已收敛为 native（保留变量用于语义清晰和后续扩展）
	const effectiveProtocol = "native"

	// 并行获取“模式说明段”和“技能说明段”，减少 system prompt 生成耗时
	const [modesSection, skillsSection] = await Promise.all([
		getModesSection(context),
		getSkillsSection(skillsManager, mode as string),
	])

	// 工具清单不内嵌在 system prompt 中（由 API metadata.tools 传递）
	const toolsCatalog = ""

	// 按固定顺序拼接系统提示词，确保提示词结构稳定可预期
	const basePrompt = `${roleDefinition}

${markdownFormattingSection()}

${getSharedToolUseSection()}${toolsCatalog}

	${getToolUseGuidelinesSection()}

${getCapabilitiesSection(cwd, shouldIncludeMcp ? mcpHub : undefined)}

${modesSection}
${skillsSection ? `\n${skillsSection}` : ""}
${getRulesSection(cwd, settings)}

${getSystemInfoSection(cwd)}

${getObjectiveSection()}

${await addCustomInstructions(baseInstructions, globalCustomInstructions || "", cwd, mode, {
	language: language ?? formatLanguage(vscode.env.language),
	rooIgnoreInstructions,
	settings,
})}`

	return basePrompt
}

/**
 * 生成最终 system prompt 的公开入口。
 *
 * 主要职责：
 * 1) 解析当前模式（含自定义模式）
 * 2) 读取该模式对应的自定义 Prompt 片段
 * 3) 调用 generatePrompt 完成统一拼接
 *
 * @returns 最终 system prompt 文本
 */
export const SYSTEM_PROMPT = async (
	context: vscode.ExtensionContext,
	cwd: string,
	supportsComputerUse: boolean,
	mcpHub?: McpHub,
	diffStrategy?: DiffStrategy,
	mode: Mode = defaultModeSlug,
	customModePrompts?: CustomModePrompts,
	customModes?: ModeConfig[],
	globalCustomInstructions?: string,
	experiments?: Record<string, boolean>,
	language?: string,
	rooIgnoreInstructions?: string,
	settings?: SystemPromptSettings,
	todoList?: TodoItem[],
	modelId?: string,
	skillsManager?: SkillsManager,
): Promise<string> => {
	if (!context) {
		throw new Error("Extension context is required for generating system prompt")
	}

	// 提取当前模式的自定义 Prompt 片段（若为空会返回 undefined）
	const promptComponent = getPromptComponent(customModePrompts, mode)

	// 解析完整模式配置：优先自定义模式，失败则回退内置模式
	const currentMode = getModeBySlug(mode, customModes) || modes.find((m) => m.slug === mode) || modes[0]

	return generatePrompt(
		context,
		cwd,
		supportsComputerUse,
		currentMode.slug,
		mcpHub,
		diffStrategy,
		promptComponent,
		customModes,
		globalCustomInstructions,
		experiments,
		language,
		rooIgnoreInstructions,
		settings,
		todoList,
		modelId,
		skillsManager,
	)
}
