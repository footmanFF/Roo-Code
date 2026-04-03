import os from "os"
import * as path from "path"
import * as fs from "fs"
import * as childProcess from "child_process"
import * as vscode from "vscode"
import ignore from "ignore"
import { arePathsEqual } from "../../utils/path"
import { getBinPath } from "../../services/ripgrep"
import { DIRS_TO_IGNORE } from "./constants"

/**
 * 目录扫描过程中的上下文对象，用于在递归扫描时传递状态信息，
 * 控制隐藏目录和忽略规则的行为。
 */
interface ScanContext {
	/** 当前目录是否是用户明确指定的目标目录（允许显示通常被忽略的隐藏目录） */
	isTargetDir: boolean
	/** 当前路径是否在一个被明确指定的隐藏目录内部（放宽内部过滤规则） */
	insideExplicitHiddenTarget: boolean
	/** 扫描操作的根目录，用于计算相对路径和 gitignore 匹配 */
	basePath: string
	/** gitignore 规则实例，用于过滤被忽略的目录 */
	ignoreInstance: ReturnType<typeof ignore>
}

/**
 * 列出目录中的文件和子目录，支持递归和非递归两种模式。
 * 同时遵守 .gitignore 规则和内置的目录忽略列表（DIRS_TO_IGNORE）。
 *
 * 递归模式下即使达到数量上限，也会强制保证一级子目录全部出现在结果中，
 * 确保 LLM 能看到项目的完整顶层结构。
 *
 * @param dirPath   要列出的目录路径（绝对或相对路径均可）
 * @param recursive 是否递归列出所有子目录中的文件
 * @param limit     返回结果的最大条数，传 0 表示不列出任何内容
 * @returns         元组 [文件路径数组, 是否已触达 limit 上限]
 */
export async function listFiles(dirPath: string, recursive: boolean, limit: number): Promise<[string[], boolean]> {
	// limit 为 0 时直接返回空结果，无需执行任何扫描
	if (limit === 0) {
		return [[], false]
	}

	// 拦截根目录和 Home 目录：这些目录文件数量极多，直接返回目录本身即可
	const specialResult = await handleSpecialDirectories(dirPath)
	if (specialResult) {
		return specialResult
	}

	// 获取 VSCode 内置的 ripgrep 二进制路径
	const rgPath = await getRipgrepPath()

	if (!recursive) {
		// 非递归模式：只列出当前目录层级的文件和直接子目录
		const files = await listFilesWithRipgrep(rgPath, dirPath, false, limit)
		const ignoreInstance = await createIgnoreInstance(dirPath)
		// 用剩余配额列出目录，避免总数超出 limit
		const remainingLimit = Math.max(0, limit - files.length)
		const directories = await listFilteredDirectories(dirPath, false, ignoreInstance, remainingLimit)
		return formatAndCombineResults(files, directories, limit)
	}

	// 递归模式：先用 ripgrep 列出文件，再用自定义逻辑列出目录（ripgrep 不输出目录）
	const files = await listFilesWithRipgrep(rgPath, dirPath, true, limit)
	const ignoreInstance = await createIgnoreInstance(dirPath)
	// 用剩余配额列出目录
	const remainingLimit = Math.max(0, limit - files.length)
	const directories = await listFilteredDirectories(dirPath, true, ignoreInstance, remainingLimit)

	// 合并文件和目录，检查是否触达上限
	const [results, limitReached] = formatAndCombineResults(files, directories, limit)

	// 触达上限时，强制将所有一级子目录插入结果，防止顶层结构缺失
	if (limitReached) {
		const firstLevelDirs = await getFirstLevelDirectories(dirPath, ignoreInstance)
		return ensureFirstLevelDirectoriesIncluded(results, firstLevelDirs, limit)
	}

	return [results, limitReached]
}

/**
 * 获取指定目录下的直接子目录列表（仅一级，不递归）。
 * 用于在结果触达上限后，确保顶层目录结构完整可见。
 *
 * @param dirPath        目标目录路径
 * @param ignoreInstance gitignore 规则实例，用于过滤被忽略的子目录
 * @returns              所有通过过滤的一级子目录路径（以 "/" 结尾）
 */
async function getFirstLevelDirectories(dirPath: string, ignoreInstance: ReturnType<typeof ignore>): Promise<string[]> {
	const absolutePath = path.resolve(dirPath)
	const directories: string[] = []

	try {
		const entries = await fs.promises.readdir(absolutePath, { withFileTypes: true })

		for (const entry of entries) {
			// 只处理真实目录，跳过符号链接（防止循环引用）
			if (entry.isDirectory() && !entry.isSymbolicLink()) {
				const fullDirPath = path.join(absolutePath, entry.name)
				// 一级子目录不是"被显式指定的目标目录"，使用普通过滤规则
				const context: ScanContext = {
					isTargetDir: false,
					insideExplicitHiddenTarget: false,
					basePath: dirPath,
					ignoreInstance,
				}
				if (shouldIncludeDirectory(entry.name, fullDirPath, context)) {
					// 目录路径统一以 "/" 结尾，与文件路径区分
					const formattedPath = fullDirPath.endsWith("/") ? fullDirPath : `${fullDirPath}/`
					directories.push(formattedPath)
				}
			}
		}
	} catch (err) {
		console.warn(`Could not read directory ${absolutePath}: ${err}`)
	}

	return directories
}

/**
 * 确保所有一级子目录都出现在最终结果中。
 * 当结果触达 limit 上限时调用：从结果末尾（通常是更深层的文件）移除若干条目，
 * 腾出空间将缺失的一级目录插入，保证 LLM 能看到完整的顶层结构。
 *
 * @param results       当前已有的结果列表（已达到 limit）
 * @param firstLevelDirs 目标目录下所有通过过滤的一级子目录
 * @param limit          结果数量上限
 * @returns              [调整后的结果列表, true（始终表示触达上限）]
 */
function ensureFirstLevelDirectoriesIncluded(
	results: string[],
	firstLevelDirs: string[],
	limit: number,
): [string[], boolean] {
	// 用 Set 加速"是否已存在"的查询
	const existingPaths = new Set(results)

	// 找出结果中缺失的一级目录
	const missingDirs = firstLevelDirs.filter((dir) => !existingPaths.has(dir))

	if (missingDirs.length === 0) {
		// 所有一级目录已在结果中，无需调整
		return [results, true]
	}

	// 从结果末尾移除与缺失目录数量等量的条目（末尾通常是树的更深层节点）
	const itemsToRemove = Math.min(missingDirs.length, results.length)
	const adjustedResults = results.slice(0, results.length - itemsToRemove)

	// 将调整后的结果按深度分为"一级条目"和"其他条目"，
	// 以便将缺失的一级目录插入到正确位置（一级区域之后）
	const resultPaths = adjustedResults.map((r) => path.resolve(r))
	// 从第一个一级目录反推父目录路径（即 basePath）
	const basePath = path.resolve(firstLevelDirs[0]).split(path.sep).slice(0, -1).join(path.sep)

	const firstLevelResults: string[] = []
	const otherResults: string[] = []

	for (let i = 0; i < adjustedResults.length; i++) {
		const resolvedPath = resultPaths[i]
		const relativePath = path.relative(basePath, resolvedPath)
		// 深度为 1 表示直接子项（一级）
		const depth = relativePath.split(path.sep).length

		if (depth === 1) {
			firstLevelResults.push(adjustedResults[i])
		} else {
			otherResults.push(adjustedResults[i])
		}
	}

	// 最终顺序：已有一级条目 → 补充的缺失一级目录 → 其余深层条目，截取到 limit
	const finalResults = [...firstLevelResults, ...missingDirs, ...otherResults].slice(0, limit)

	return [finalResults, true]
}

/**
 * 处理特殊目录（根目录和用户 Home 目录），这两个目录文件数量极大，
 * 直接列出会产生无意义的海量输出，因此拦截并只返回目录本身。
 *
 * @param dirPath 目标目录路径
 * @returns 若是特殊目录，返回 [只含目录本身的数组, false]；否则返回 null 表示无需特殊处理
 */
async function handleSpecialDirectories(dirPath: string): Promise<[string[], boolean] | null> {
	const absolutePath = path.resolve(dirPath)

	// 拦截根目录（Windows 下为盘符如 C:\，Unix 下为 /）
	const root = process.platform === "win32" ? path.parse(absolutePath).root : "/"
	const isRoot = arePathsEqual(absolutePath, root)
	if (isRoot) {
		return [[root], false]
	}

	// 拦截用户 Home 目录（如 /Users/xxx 或 C:\Users\xxx）
	const homeDir = os.homedir()
	const isHomeDir = arePathsEqual(absolutePath, homeDir)
	if (isHomeDir) {
		return [[homeDir], false]
	}

	return null
}

/**
 * 获取 VSCode 内置 ripgrep 二进制的完整路径。
 * 找不到时抛出错误（说明 VSCode 环境异常）。
 *
 * @returns ripgrep 二进制的绝对路径
 * @throws  找不到 ripgrep 时抛出 Error
 */
async function getRipgrepPath(): Promise<string> {
	const vscodeAppRoot = vscode.env.appRoot
	const rgPath = await getBinPath(vscodeAppRoot)

	if (!rgPath) {
		throw new Error("Could not find ripgrep binary")
	}

	return rgPath
}

/**
 * 调用 ripgrep 列出目录中的文件路径，并将相对路径转换为绝对路径返回。
 * 注意：ripgrep 只输出文件，不输出目录，目录由 listFilteredDirectories 单独处理。
 *
 * @param rgPath    ripgrep 二进制路径
 * @param dirPath   要扫描的目录
 * @param recursive 是否递归扫描
 * @param limit     最多返回的文件数
 * @returns         文件的绝对路径列表
 */
async function listFilesWithRipgrep(
	rgPath: string,
	dirPath: string,
	recursive: boolean,
	limit: number,
): Promise<string[]> {
	const rgArgs = buildRipgrepArgs(dirPath, recursive)

	// ripgrep 输出的是相对于搜索目录的相对路径
	const relativePaths = await execRipgrep(rgPath, rgArgs, limit)

	// 转换为绝对路径，dirPath 只解析一次避免重复计算
	const absolutePath = path.resolve(dirPath)
	return relativePaths.map((relativePath) => path.resolve(absolutePath, relativePath))
}

/**
 * 根据是否递归，构建 ripgrep 的命令行参数列表。
 *
 * 基础参数含义：
 *   --files   只列出文件名，不做内容匹配
 *   --hidden  包含隐藏文件（以 "." 开头的文件）
 *   --follow  跟随符号链接
 *
 * @param dirPath   要搜索的目录
 * @param recursive 是否递归搜索
 * @returns         完整的 rg 参数数组
 */
function buildRipgrepArgs(dirPath: string, recursive: boolean): string[] {
	const args = ["--files", "--hidden", "--follow"]

	if (recursive) {
		return [...args, ...buildRecursiveArgs(dirPath), dirPath]
	} else {
		return [...args, ...buildNonRecursiveArgs(), dirPath]
	}
}

/**
 * 构建递归模式下的 ripgrep 专属参数。
 *
 * 核心策略：
 * - 默认情况下 ripgrep 自动遵守 .gitignore，无需额外配置
 * - 若目标路径本身是隐藏目录或在忽略列表中，则关闭 VCS/ignore 规则，强制显示所有文件
 * - 对 DIRS_TO_IGNORE 中的目录追加 glob 排除规则，避免扫描无意义的大型目录（如 node_modules）
 *
 * @param dirPath 要递归扫描的目录路径
 * @returns       递归模式专属的 rg 参数数组（不含基础参数和目录路径）
 */
function buildRecursiveArgs(dirPath: string): string[] {
	const args: string[] = []

	// 规范化路径后按分隔符拆分，判断路径中是否含有隐藏目录组件（以 "." 开头）
	// 过滤空字符串以处理末尾斜杠、重复分隔符等边界情况
	const normalizedPath = path.normalize(dirPath)
	const pathParts = normalizedPath.split(path.sep).filter((part) => part.length > 0)
	const isTargetingHiddenDir = pathParts.some((part) => part.startsWith("."))

	// 判断目标目录名是否在内置忽略列表中（如 "temp"、"dist" 等）
	const targetDirName = path.basename(dirPath)
	const isTargetInIgnoreList = DIRS_TO_IGNORE.includes(targetDirName)

	// 显式指定隐藏目录或忽略列表中的目录时，需要绕过默认的 ignore 规则
	if (isTargetingHiddenDir || isTargetInIgnoreList) {
		// --no-ignore-vcs：禁用 .gitignore 等 VCS 忽略规则
		// --no-ignore：禁用 .ignore 文件的规则
		args.push("--no-ignore-vcs")
		args.push("--no-ignore")
		// 显式包含根目录下的文件和所有子目录文件
		args.push("-g", "*")
		args.push("-g", "**/*")
	}

	// 为 DIRS_TO_IGNORE 中的每一项追加 glob 排除规则
	for (const dir of DIRS_TO_IGNORE) {
		if (dir === ".*") {
			if (!isTargetingHiddenDir) {
				// 未显式指定隐藏目录时，排除所有隐藏目录（如 .git、.cache 等）
				args.push("-g", `!**/.*/**`)
			}
			// 显式指定隐藏目录时跳过此规则，允许其内容被列出
			continue
		}

		if (dir === targetDirName && isTargetInIgnoreList) {
			// 目标目录本身在忽略列表中时，不添加排除规则（要展示其内容）
			// 但嵌套的同名子目录仍会被其他规则过滤
			continue
		}

		// 其他情况：全局排除该目录名下的所有内容
		args.push("-g", `!**/${dir}/**`)
	}

	return args
}

/**
 * 构建非递归模式下的 ripgrep 专属参数。
 *
 * 非递归模式只列出当前目录层级的直接子文件：
 * - 使用 --maxdepth 1 限制深度为 1
 * - 隐藏目录（".*"）不排除文件本身，由目录扫描逻辑单独控制其可见性
 * - 其他忽略目录只排除其直接子项（不用 ** 通配）
 *
 * @returns 非递归模式专属的 rg 参数数组（不含基础参数和目录路径）
 */
function buildNonRecursiveArgs(): string[] {
	const args: string[] = []

	// 只匹配当前目录层级，-g "*" 配合 --maxdepth 1 实现
	args.push("-g", "*")
	// ripgrep 使用 --maxdepth（不是 --max-depth）
	args.push("--maxdepth", "1")

	for (const dir of DIRS_TO_IGNORE) {
		if (dir === ".*") {
			// 非递归模式下，隐藏目录自身可以出现在列表中（用户能看到有哪些隐藏目录），
			// 但 --maxdepth 1 天然阻止了其内容被列出，无需额外排除
			continue
		} else {
			// 排除忽略目录本身及其直接子内容（非递归无需 **）
			args.push("-g", `!${dir}`)
			args.push("-g", `!${dir}/**`)
		}
	}

	return args
}

/**
 * 创建 gitignore 规则实例，用于目录过滤。
 * 从目标目录向上逐级查找所有 .gitignore 文件，合并其中的规则，
 * 从而实现与 git 一致的忽略行为（根目录的规则最先加载，子目录规则后加载，优先级更高）。
 *
 * @param dirPath 要扫描的目录路径
 * @returns       已加载所有 .gitignore 规则的 ignore 实例
 */
async function createIgnoreInstance(dirPath: string): Promise<ReturnType<typeof ignore>> {
	const ignoreInstance = ignore()
	const absolutePath = path.resolve(dirPath)

	// 从目标目录向上递归查找所有 .gitignore 文件（按从根到子的顺序返回）
	const gitignoreFiles = await findGitignoreFiles(absolutePath)

	// 将所有 .gitignore 的规则依次加入 ignore 实例
	for (const gitignoreFile of gitignoreFiles) {
		try {
			const content = await fs.promises.readFile(gitignoreFile, "utf8")
			ignoreInstance.add(content)
		} catch (err) {
			// 读取失败时跳过（如权限问题），不影响整体流程
			console.warn(`Could not read .gitignore at ${gitignoreFile}: ${err}`)
		}
	}

	// .gitignore 文件本身也加入忽略列表，避免出现在结果中
	ignoreInstance.add(".gitignore")

	return ignoreInstance
}

/**
 * 从指定目录向上遍历目录树，收集所有 .gitignore 文件的路径。
 * 最终以"根目录优先"的顺序返回，确保父级规则先被加载、子级规则后覆盖，
 * 与 git 的规则优先级一致。
 *
 * @param startPath 起始目录的绝对路径
 * @returns         按从根到子顺序排列的 .gitignore 文件路径列表
 */
async function findGitignoreFiles(startPath: string): Promise<string[]> {
	const gitignoreFiles: string[] = []
	let currentPath = startPath

	// 向上逐级检查，直到到达文件系统根目录（path.dirname(root) === root）
	while (currentPath && currentPath !== path.dirname(currentPath)) {
		const gitignorePath = path.join(currentPath, ".gitignore")

		try {
			// access 成功说明文件存在，加入列表
			await fs.promises.access(gitignorePath)
			gitignoreFiles.push(gitignorePath)
		} catch {
			// 当前层没有 .gitignore，继续向上
		}

		const parentPath = path.dirname(currentPath)
		if (parentPath === currentPath) {
			break // 已到达根目录，终止循环
		}
		currentPath = parentPath
	}

	// 反转顺序：根目录的 .gitignore 最先加载，层级越深的规则越晚加载（优先级越高）
	return gitignoreFiles.reverse()
}

/**
 * 列出目录下的子目录列表（ripgrep 只输出文件，目录需单独处理）。
 * 支持递归和非递归模式，并根据 gitignore 规则、内置忽略列表和隐藏目录策略进行过滤。
 *
 * @param dirPath        要扫描的根目录
 * @param recursive      是否递归扫描子目录
 * @param ignoreInstance gitignore 规则实例
 * @param limit          最多返回的目录数量，不传则不限制
 * @returns              通过过滤的目录绝对路径列表（以 "/" 结尾）
 */
async function listFilteredDirectories(
	dirPath: string,
	recursive: boolean,
	ignoreInstance: ReturnType<typeof ignore>,
	limit?: number,
): Promise<string[]> {
	const absolutePath = path.resolve(dirPath)
	const directories: string[] = []
	let dirCount = 0
	// 未传 limit 时使用最大安全整数，相当于不限制
	const effectiveLimit = limit ?? Number.MAX_SAFE_INTEGER

	// 仅当目标目录本身是隐藏目录（如 .roo-memory）时，才将其标记为 isTargetDir，
	// 放宽其直接子目录的过滤规则。普通递归扫描不走此分支。
	const isExplicitHiddenTarget = path.basename(absolutePath).startsWith(".")

	const initialContext: ScanContext = {
		isTargetDir: isExplicitHiddenTarget,
		insideExplicitHiddenTarget: isExplicitHiddenTarget,
		basePath: dirPath,
		ignoreInstance,
	}

	/**
	 * 递归扫描单个目录，收集其中的子目录。
	 * @param currentPath 当前正在扫描的目录绝对路径
	 * @param context     当前扫描上下文（控制隐藏目录的过滤行为）
	 * @returns           true 表示已触达数量上限，外层应停止继续扫描
	 */
	async function scanDirectory(currentPath: string, context: ScanContext): Promise<boolean> {
		if (dirCount >= effectiveLimit) {
			return true
		}

		try {
			const entries = await fs.promises.readdir(currentPath, { withFileTypes: true })

			for (const entry of entries) {
				if (dirCount >= effectiveLimit) {
					return true
				}

				// 只处理真实目录，跳过符号链接（防止循环引用导致无限递归）
				if (entry.isDirectory() && !entry.isSymbolicLink()) {
					const dirName = entry.name
					const fullDirPath = path.join(currentPath, dirName)

					// 扫描过程中遇到的子目录不是"被显式指定的目标"，isTargetDir 置为 false
					const subdirContext: ScanContext = {
						...context,
						isTargetDir: false,
					}

					if (shouldIncludeDirectory(dirName, fullDirPath, subdirContext)) {
						// 目录路径以 "/" 结尾，与文件路径区分
						const formattedPath = fullDirPath.endsWith("/") ? fullDirPath : `${fullDirPath}/`
						directories.push(formattedPath)
						dirCount++

						if (dirCount >= effectiveLimit) {
							return true
						}
					}

					const isHiddenDir = dirName.startsWith(".")

					// 判断是否应该递归进入此子目录：
					// - 在显式指定隐藏目录的内部：只阻止 CRITICAL_IGNORE_PATTERNS 中的目录
					// - 普通情况：阻止 DIRS_TO_IGNORE 中的目录
					let shouldRecurseIntoDir = true
					if (context.insideExplicitHiddenTarget) {
						shouldRecurseIntoDir = !CRITICAL_IGNORE_PATTERNS.has(dirName)
					} else {
						shouldRecurseIntoDir = !isDirectoryExplicitlyIgnored(dirName)
					}

					// 综合判断是否递归：
					// 1. 必须是递归模式
					// 2. 目录本身未被忽略
					// 3. 不能是隐藏目录（除非是被显式指定的目标或已在显式隐藏目标内部）
					const shouldRecurse =
						recursive &&
						shouldRecurseIntoDir &&
						!(
							isHiddenDir &&
							DIRS_TO_IGNORE.includes(".*") &&
							!context.isTargetDir &&
							!context.insideExplicitHiddenTarget
						)

					if (shouldRecurse) {
						// 进入隐藏子目录时，更新 insideExplicitHiddenTarget 标志，
						// 使该隐藏目录内部的子目录也能被正常列出
						const newInsideExplicitHiddenTarget =
							context.insideExplicitHiddenTarget || (isHiddenDir && context.isTargetDir)
						const newContext: ScanContext = {
							...context,
							isTargetDir: false,
							insideExplicitHiddenTarget: newInsideExplicitHiddenTarget,
						}
						const limitReached = await scanDirectory(fullDirPath, newContext)
						if (limitReached) {
							return true
						}
					}
				}
			}
		} catch (err) {
			// 无法读取目录时（如权限问题）跳过，不中断整体扫描
			console.warn(`Could not read directory ${currentPath}: ${err}`)
		}

		return false
	}

	await scanDirectory(absolutePath, initialContext)

	return directories
}

/**
 * 即使在显式指定的隐藏目录内部，也必须始终被忽略的关键目录集合。
 * 包含依赖包目录、版本控制目录、Python 缓存/虚拟环境目录等，
 * 这些目录内容庞大且对代码检索没有价值。
 */
const CRITICAL_IGNORE_PATTERNS = new Set(["node_modules", ".git", "__pycache__", "venv", "env"])

/**
 * 检查目录名是否匹配给定的忽略模式列表。
 * 支持精确名称匹配和路径前缀匹配（如 "foo/bar" 模式匹配名为 "foo" 的目录）。
 *
 * @param dirName  要检查的目录名
 * @param patterns 忽略模式列表
 * @returns        true 表示目录名命中了某个忽略模式
 */
function matchesIgnorePattern(dirName: string, patterns: string[]): boolean {
	for (const pattern of patterns) {
		if (pattern === dirName || (pattern.includes("/") && pattern.split("/")[0] === dirName)) {
			return true
		}
	}
	return false
}

/**
 * 判断目录是否应被 gitignore 规则过滤掉。
 * 同时检查不带斜杠和带斜杠（目录专用）两种路径形式，确保不遗漏匹配。
 *
 * @param fullDirPath    目录的绝对路径
 * @param basePath       计算相对路径所用的基准目录
 * @param ignoreInstance 已加载规则的 ignore 实例
 * @returns              true 表示该目录被 gitignore 规则忽略
 */
function isIgnoredByGitignore(
	fullDirPath: string,
	basePath: string,
	ignoreInstance: ReturnType<typeof ignore>,
): boolean {
	// 将绝对路径转为相对路径，并统一为 "/" 分隔符（兼容 Windows）
	const relativePath = path.relative(basePath, fullDirPath)
	const normalizedPath = relativePath.replace(/\\/g, "/")
	// 分别以"路径"和"路径/"两种形式检查，因为 .gitignore 中目录专用规则通常带斜杠
	return ignoreInstance.ignores(normalizedPath) || ignoreInstance.ignores(normalizedPath + "/")
}

/**
 * 判断"被显式指定的目标目录"是否应出现在结果中。
 * 对显式目标目录放宽规则：忽略 ".*" 模式（允许隐藏目录本身），
 * 但仍然过滤其他内置忽略规则（如 node_modules）。
 *
 * @param dirName 目录名
 * @returns       true 表示应包含此目录
 */
function shouldIncludeTargetDirectory(dirName: string): boolean {
	// 显式目标目录跳过 ".*" 隐藏目录过滤，只检查其他忽略规则
	const nonHiddenIgnorePatterns = DIRS_TO_IGNORE.filter((pattern) => pattern !== ".*")
	return !matchesIgnorePattern(dirName, nonHiddenIgnorePatterns)
}

/**
 * 判断"位于显式指定隐藏目录内部"的子目录是否应出现在结果中。
 * 在该场景下放宽大部分过滤规则，仅阻止 CRITICAL_IGNORE_PATTERNS 和 gitignore 规则命中的目录。
 *
 * @param dirName     子目录名
 * @param fullDirPath 子目录绝对路径
 * @param context     当前扫描上下文
 * @returns           true 表示应包含此目录
 */
function shouldIncludeInsideHiddenTarget(dirName: string, fullDirPath: string, context: ScanContext): boolean {
	// 即使在隐藏目录内部，关键的大型/无意义目录仍需屏蔽
	if (CRITICAL_IGNORE_PATTERNS.has(dirName)) {
		return false
	}

	// 尊重 gitignore 规则
	return !isIgnoredByGitignore(fullDirPath, context.basePath, context.ignoreInstance)
}

/**
 * 判断普通目录（非隐藏目标、非隐藏目标内部）是否应出现在结果中。
 * 同时检查内置忽略列表（排除 ".*" 由调用方单独处理）和 gitignore 规则。
 *
 * @param dirName     目录名
 * @param fullDirPath 目录绝对路径
 * @param context     当前扫描上下文
 * @returns           true 表示应包含此目录
 */
function shouldIncludeRegularDirectory(dirName: string, fullDirPath: string, context: ScanContext): boolean {
	// 检查内置忽略列表，".*" 模式由外层的隐藏目录判断逻辑处理，此处跳过
	const nonHiddenIgnorePatterns = DIRS_TO_IGNORE.filter((pattern) => pattern !== ".*")
	if (matchesIgnorePattern(dirName, nonHiddenIgnorePatterns)) {
		return false
	}

	// 检查 gitignore 规则
	return !isIgnoredByGitignore(fullDirPath, context.basePath, context.ignoreInstance)
}

/**
 * 根据当前上下文决定某个目录是否应出现在结果中。
 * 这是目录包含决策的统一入口，根据三种场景分别路由：
 * 1. 被显式指定的目标目录（如用户直接指定 .roo-memory）
 * 2. 位于显式隐藏目标目录内部的子目录
 * 3. 普通目录（默认过滤规则）
 *
 * @param dirName     目录名
 * @param fullDirPath 目录绝对路径
 * @param context     当前扫描上下文
 * @returns           true 表示应包含此目录
 */
function shouldIncludeDirectory(dirName: string, fullDirPath: string, context: ScanContext): boolean {
	if (context.isTargetDir) {
		// 显式目标目录：放宽隐藏目录限制，允许 .roo-memory 等出现
		return shouldIncludeTargetDirectory(dirName)
	}

	if (context.insideExplicitHiddenTarget) {
		// 在显式隐藏目标内部：只阻止关键忽略目录和 gitignore 命中项
		return shouldIncludeInsideHiddenTarget(dirName, fullDirPath, context)
	}

	// 普通情况：应用完整过滤规则
	return shouldIncludeRegularDirectory(dirName, fullDirPath, context)
}

/**
 * 检查目录是否在内置的显式忽略列表（DIRS_TO_IGNORE）中。
 * 注意：此函数专门用于控制"是否递归进入"某个目录，与 shouldIncludeDirectory 的职责不同。
 * ".*" 模式被跳过，其对应的隐藏目录可见性由 shouldIncludeDirectory 中的逻辑单独决定。
 *
 * @param dirName 目录名
 * @returns       true 表示该目录在显式忽略列表中，不应被递归遍历
 */
function isDirectoryExplicitlyIgnored(dirName: string): boolean {
	for (const pattern of DIRS_TO_IGNORE) {
		// 精确名称匹配
		if (pattern === dirName) {
			return true
		}

		// ".*" 作为特殊模式，不参与此处的递归判断（由隐藏目录专用逻辑处理）
		if (pattern === ".*") {
			continue
		}

		// 形如 "foo/bar" 的路径模式：只要目录名匹配第一个路径段即视为忽略
		if (pattern.includes("/")) {
			const pathParts = pattern.split("/")
			if (pathParts[0] === dirName) {
				return true
			}
		}
	}

	return false
}

/**
 * 合并文件列表和目录列表，去重、排序并截断到上限。
 * 排序规则：目录优先于文件，同类型路径按字母顺序排列。
 *
 * @param files       文件路径数组（由 ripgrep 输出）
 * @param directories 目录路径数组（由 listFilteredDirectories 输出，以 "/" 结尾）
 * @param limit       结果数量上限
 * @returns           [截断后的路径数组, 是否触达上限]
 */
function formatAndCombineResults(files: string[], directories: string[], limit: number): [string[], boolean] {
	// 合并文件和目录（目录放前面，排序后可能调整顺序）
	const allPaths = [...directories, ...files]

	// 去重：避免目录同时出现在 ripgrep 结果和目录扫描结果中
	const uniquePathsSet = new Set(allPaths)
	const uniquePaths = Array.from(uniquePathsSet)

	// 排序：目录（以 "/" 结尾）排在文件前面，同类按字母顺序
	uniquePaths.sort((a: string, b: string) => {
		const aIsDir = a.endsWith("/")
		const bIsDir = b.endsWith("/")

		if (aIsDir && !bIsDir) return -1
		if (!aIsDir && bIsDir) return 1
		return a.localeCompare(b)
	})

	const trimmedPaths = uniquePaths.slice(0, limit)
	// 若截断后长度等于 limit，说明可能还有更多结果未显示
	return [trimmedPaths, trimmedPaths.length >= limit]
}

/**
 * 以子进程方式执行 ripgrep，并流式收集文件路径列表。
 * - 达到 limit 上限时立即终止子进程，避免无谓的 I/O 开销
 * - 设有 10 秒超时保护，超时后返回已收集的部分结果
 * - 非零退出码（除 SIGTERM=143）仅打印警告，不抛出错误
 *
 * @param rgPath ripgrep 二进制文件的绝对路径
 * @param args   传递给 ripgrep 的完整参数列表（最后一项为目标目录）
 * @param limit  最多返回的文件路径数量
 * @returns      文件路径字符串数组（ripgrep 输出的相对路径）
 */
async function execRipgrep(rgPath: string, args: string[], limit: number): Promise<string[]> {
	return new Promise((resolve, reject) => {
		const rgProcess = childProcess.spawn(rgPath, args)
		let output = ""
		let results: string[] = []

		// 超时保护：10 秒内未完成则强制终止并返回部分结果
		const timeoutId = setTimeout(() => {
			rgProcess.kill()
			console.warn("ripgrep timed out, returning partial results")
			resolve(results.slice(0, limit))
		}, 10_000)

		// 流式处理 stdout，收到足够数量的结果后立即终止进程
		rgProcess.stdout.on("data", (data) => {
			output += data.toString()
			processRipgrepOutput()

			if (results.length >= limit) {
				rgProcess.kill()
				clearTimeout(timeoutId)
			}
		})

		rgProcess.stderr.on("data", (data) => {
			console.error(`ripgrep stderr: ${data}`)
		})

		rgProcess.on("close", (code) => {
			clearTimeout(timeoutId)

			// 处理缓冲区中剩余的最后一行（isFinal=true 时不保留未完成行）
			processRipgrepOutput(true)

			// 143 是 SIGTERM 的退出码（主动 kill），属于正常情况，不需警告
			if (code !== 0 && code !== null && code !== 143) {
				console.warn(`ripgrep process exited with code ${code}, returning partial results`)
			}

			resolve(results.slice(0, limit))
		})

		rgProcess.on("error", (error) => {
			clearTimeout(timeoutId)
			reject(new Error(`ripgrep process error: ${error.message}`))
		})

		/**
		 * 处理输出缓冲区，将完整行解析为路径并加入 results。
		 * @param isFinal 是否为最后一次调用（进程已关闭），true 时不保留不完整的最后一行
		 */
		function processRipgrepOutput(isFinal = false) {
			const lines = output.split("\n")

			if (!isFinal) {
				// 最后一行可能是不完整的，暂存到 output 等待下次数据到来
				output = lines.pop() || ""
			} else {
				output = ""
			}

			for (const line of lines) {
				if (line.trim() && results.length < limit) {
					results.push(line)
				} else if (results.length >= limit) {
					break
				}
			}
		}
	})
}
