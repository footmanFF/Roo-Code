import * as childProcess from "child_process"
import * as path from "path"
import * as readline from "readline"

import * as vscode from "vscode"

import { RooIgnoreController } from "../../core/ignore/RooIgnoreController"
import { fileExistsAtPath } from "../../utils/fs"
/*
本文件基于 ripgrep 实现文件内容的正则检索功能。
参考自：https://github.com/DiscreteTom/vscode-ripgrep-utils

主要组成部分：
1. getBinPath：在 VSCode 安装目录中定位 ripgrep 二进制文件。
2. execRipgrep：执行 ripgrep 子进程并返回原始输出。
3. regexSearchFiles：对外暴露的主函数，执行正则搜索并返回格式化结果。
   - 参数：
     * cwd：工作区根目录（用于将绝对路径转换为相对路径）
     * directoryPath：要搜索的目录
     * regex：正则表达式（Rust regex 语法）
     * filePattern：可选的 glob 文件类型过滤（默认搜索全部文件）
   - 返回值：包含匹配结果和上下文的格式化字符串

搜索结果包含：
- 相对文件路径
- 每个匹配行前后各 1 行上下文
- 行号与内容通过 " | " 分隔，便于阅读

使用示例：
const results = await regexSearchFiles('/path/to/cwd', '/path/to/search', 'TODO:', '*.ts');

rel/path/to/app.ts
│----
│function processData(data: any) {
│  // Some processing logic here
│  // TODO: Implement error handling
│  return processedData;
│}
│----

rel/path/to/helper.ts
│----
│  let result = 0;
│  for (let i = 0; i < input; i++) {
│    // TODO: Optimize this function for performance
│    result += Math.pow(i, 2);
│  }
│----
*/

// 跨平台判断：Windows 下 ripgrep 二进制名为 rg.exe，其他平台为 rg
const isWindows = process.platform.startsWith("win")
const binName = isWindows ? "rg.exe" : "rg"

// 单个文件的搜索结果：文件路径 + 该文件内所有匹配分组
interface SearchFileResult {
	file: string             // 文件的绝对路径
	searchResults: SearchResult[]  // 该文件中的匹配结果列表（每个连续匹配块为一组）
}

// 一个连续匹配块，包含若干相邻行（匹配行 + 上下文行）
interface SearchResult {
	lines: SearchLineResult[]
}

// 单行信息
interface SearchLineResult {
	line: number      // 行号（1-based）
	text: string      // 行文本内容（超长会被截断）
	isMatch: boolean  // true 表示匹配行，false 表示上下文行
	column?: number   // 匹配列的字节偏移量（仅匹配行有值）
}

// 最多返回 300 条匹配结果，避免输出过大撑爆上下文
const MAX_RESULTS = 300
// 单行最大字符数，超过则截断并追加 [truncated...]
const MAX_LINE_LENGTH = 500

/**
 * 截断超长行，防止单行内容过长撑爆 LLM 上下文。
 * @param line 原始行文本
 * @param maxLength 最大允许长度，默认 MAX_LINE_LENGTH
 */
export function truncateLine(line: string, maxLength: number = MAX_LINE_LENGTH): string {
	return line.length > maxLength ? line.substring(0, maxLength) + " [truncated...]" : line
}

/**
 * 在 VSCode 安装目录中查找 ripgrep 二进制的路径。
 * VSCode 内置了 ripgrep，分布在以下几个可能的位置（普通安装 / asar 解压目录）。
 * 按优先级依次检查，返回第一个存在的路径。
 */
export async function getBinPath(vscodeAppRoot: string): Promise<string | undefined> {
	const checkPath = async (pkgFolder: string) => {
		const fullPath = path.join(vscodeAppRoot, pkgFolder, binName)
		return (await fileExistsAtPath(fullPath)) ? fullPath : undefined
	}

	return (
		(await checkPath("node_modules/@vscode/ripgrep/bin/")) ||
		(await checkPath("node_modules/vscode-ripgrep/bin")) ||
		(await checkPath("node_modules.asar.unpacked/vscode-ripgrep/bin/")) ||
		(await checkPath("node_modules.asar.unpacked/@vscode/ripgrep/bin/"))
	)
}

/**
 * 执行 ripgrep 子进程，流式读取输出并在达到行数上限时主动终止进程。
 *
 * 使用 readline 逐行读取而非一次性读取全部 stdout，
 * 这是 ripgrep 作者推荐的跨平台限制输出的方式（替代 `head` 命令）。
 *
 * @param bin  ripgrep 二进制路径
 * @param args 传给 rg 的命令行参数列表
 */
async function execRipgrep(bin: string, args: string[]): Promise<string> {
	return new Promise((resolve, reject) => {
		const rgProcess = childProcess.spawn(bin, args)

		// crlfDelay: Infinity 确保 \r\n 被视为单个换行符，跨平台行为一致
		const rl = readline.createInterface({
			input: rgProcess.stdout,
			crlfDelay: Infinity,
		})

		let output = ""
		let lineCount = 0
		// 每条结果最多 5 行（1 匹配行 + 前后各 1 上下文行 + 分隔符），
		// 因此行数上限 = MAX_RESULTS * 5
		const maxLines = MAX_RESULTS * 5

		rl.on("line", (line) => {
			if (lineCount < maxLines) {
				output += line + "\n"
				lineCount++
			} else {
				// 达到行数上限，主动关闭 readline 并终止子进程，避免无限等待
				rl.close()
				rgProcess.kill()
			}
		})

		let errorOutput = ""
		rgProcess.stderr.on("data", (data) => {
			errorOutput += data.toString()
		})

		rl.on("close", () => {
			// stderr 有内容说明 rg 报错（如路径不存在），转为 reject
			if (errorOutput) {
				reject(new Error(`ripgrep process error: ${errorOutput}`))
			} else {
				resolve(output)
			}
		})

		rgProcess.on("error", (error) => {
			reject(new Error(`ripgrep process error: ${error.message}`))
		})
	})
}

/**
 * 精准检索入口：调用 ripgrep 在指定目录内执行正则搜索，返回格式化的结果字符串。
 *
 * @param cwd              工作区根目录，用于将绝对路径转为相对路径显示
 * @param directoryPath    实际搜索的目录（绝对路径）
 * @param regex            正则表达式（Rust regex 语法）
 * @param filePattern      可选的 glob 文件过滤，如 "*.ts"；不传则搜索所有文件
 * @param rooIgnoreController  可选的忽略规则控制器，过滤 .rooignore 中列出的文件
 */
export async function regexSearchFiles(
	cwd: string,
	directoryPath: string,
	regex: string,
	filePattern?: string,
	rooIgnoreController?: RooIgnoreController,
): Promise<string> {
	const vscodeAppRoot = vscode.env.appRoot
	const rgPath = await getBinPath(vscodeAppRoot)

	if (!rgPath) {
		throw new Error("Could not find ripgrep binary")
	}

	// --json：结构化输出，每行一个 JSON 事件（begin/match/context/end）
	// -e：指定正则表达式
	const args = ["--json", "-e", regex]

	// 仅在明确指定文件类型时才添加 --glob。
	// 若传入 --glob "*" 会覆盖 .gitignore 的过滤行为，因此无 pattern 时省略该参数。
	if (filePattern) {
		args.push("--glob", filePattern)
	}

	// --context 1：匹配行前后各保留 1 行上下文
	// --no-messages：抑制权限错误等非关键警告信息
	args.push("--context", "1", "--no-messages", directoryPath)

	let output: string
	try {
		output = await execRipgrep(rgPath, args)
	} catch (error) {
		console.error("Error executing ripgrep:", error)
		return "No results found"
	}

	const results: SearchFileResult[] = []
	let currentFile: SearchFileResult | null = null

	// 逐行解析 ripgrep 的 JSON 输出，rg --json 每行输出一个事件：
	// - begin：开始处理新文件
	// - match：命中正则的行
	// - context：匹配行的上下文行
	// - end：当前文件处理结束
	output.split("\n").forEach((line) => {
		if (line) {
			try {
				const parsed = JSON.parse(line)
				if (parsed.type === "begin") {
					// 新文件开始，初始化当前文件的结果容器
					currentFile = {
						file: parsed.data.path.text.toString(),
						searchResults: [],
					}
				} else if (parsed.type === "end") {
					// 当前文件结束，将结果推入总列表并重置
					results.push(currentFile as SearchFileResult)
					currentFile = null
				} else if ((parsed.type === "match" || parsed.type === "context") && currentFile) {
					const line = {
						line: parsed.data.line_number,
						text: truncateLine(parsed.data.lines.text),
						isMatch: parsed.type === "match",
						// 只有 match 行才有列偏移信息
						...(parsed.type === "match" && { column: parsed.data.absolute_offset }),
					}

					const lastResult = currentFile.searchResults[currentFile.searchResults.length - 1]
					if (lastResult?.lines.length > 0) {
						const lastLine = lastResult.lines[lastResult.lines.length - 1]

						// 若当前行与上一行相邻（行号差 ≤ 1），则合并到同一个匹配块中
						if (parsed.data.line_number <= lastLine.line + 1) {
							lastResult.lines.push(line)
						} else {
							// 行号不连续，说明这是一个新的独立匹配块
							currentFile.searchResults.push({
								lines: [line],
							})
						}
					} else {
						// 当前文件的第一行匹配，创建第一个匹配块
						currentFile.searchResults.push({
							lines: [line],
						})
					}
				}
			} catch (error) {
				console.error("Error parsing ripgrep output:", error)
			}
		}
	})

	// 若配置了 .rooignore，过滤掉被忽略的文件路径
	const filteredResults = rooIgnoreController
		? results.filter((result) => rooIgnoreController.validateAccess(result.file))
		: results

	return formatResults(filteredResults, cwd)
}

/**
 * 将结构化的检索结果格式化为纯文本字符串，供 LLM 阅读。
 *
 * 输出格式：
 *   # 相对文件路径
 *    12 | 匹配行内容
 *    13 | 下一行内容
 *   ----
 *
 * @param fileResults 所有文件的匹配结果
 * @param cwd         工作区根目录，用于生成相对路径
 */
function formatResults(fileResults: SearchFileResult[], cwd: string): string {
	const groupedResults: { [key: string]: SearchResult[] } = {}

	const totalResults = fileResults.reduce((sum, file) => sum + file.searchResults.length, 0)
	let output = ""

	// 超过上限时提示用户缩小搜索范围
	if (totalResults >= MAX_RESULTS) {
		output += `Showing first ${MAX_RESULTS} of ${MAX_RESULTS}+ results. Use a more specific search if necessary.\n\n`
	} else {
		output += `Found ${totalResults === 1 ? "1 result" : `${totalResults.toLocaleString()} results`}.\n\n`
	}

	// 将绝对路径转换为相对路径，并按文件分组（截取前 MAX_RESULTS 条）
	fileResults.slice(0, MAX_RESULTS).forEach((file) => {
		const relativeFilePath = path.relative(cwd, file.file)
		if (!groupedResults[relativeFilePath]) {
			groupedResults[relativeFilePath] = []
			groupedResults[relativeFilePath].push(...file.searchResults)
		}
	})

	// 按文件输出，每个匹配块以 ---- 分隔
	for (const [filePath, fileResults] of Object.entries(groupedResults)) {
		output += `# ${filePath.toPosix()}\n`

		fileResults.forEach((result) => {
			if (result.lines.length > 0) {
				result.lines.forEach((line) => {
					// 行号右对齐，固定 3 位宽度，与内容用 " | " 分隔
					const lineNumber = String(line.line).padStart(3, " ")
					output += `${lineNumber} | ${line.text.trimEnd()}\n`
				})
				output += "----\n"
			}
		})

		output += "\n"
	}

	return output.trim()
}
