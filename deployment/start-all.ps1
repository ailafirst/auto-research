<#
.SYNOPSIS
    启动 Deep Research Agent 的全部服务。

.DESCRIPTION
    部署拓扑 —— 只有模型服务留在宿主机，其余一切在 Docker 里：

        宿主机   model-server :8100     需要 GPU 和约 7.1GB 权重，不适合进容器
        容器     nginx :80  →  api  →  worker × N
                              ↘  mysql / redis / qdrant

    首次运行会自动生成 deployment/docker/.env，并为 MySQL / Redis / Qdrant 各写入
    一串随机口令；该文件不入版本库，重复运行不会覆盖已有口令。

    脚本会先把模型服务拉起来（冷启动约 100 秒），趁它装载权重的同时构建并启动容器，
    最后统一等待就绪并逐项校验依赖。

.EXAMPLE
    .\deployment\start-all.ps1

.EXAMPLE
    .\deployment\start-all.ps1 -WorkerCount 4 -PythonExe "D:\conda\envs\deepresearch\python.exe"
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 8)]
    [int]$WorkerCount = 2,

    [string]$PythonExe = "",

    # 模型服务的监听地址。默认 0.0.0.0 是必需的：容器经 host.docker.internal 连过来
    # 走的是宿主机网卡地址而不是回环，绑 127.0.0.1 时容器一定连不上。代价是 :8100
    # 对局域网可见且模型服务自身无鉴权 —— 请按 README「把模型服务挡在防火墙后面」
    # 加一条入站规则。只在宿主机内自测时可以传 127.0.0.1。
    [string]$ModelServerHost = "0.0.0.0",

    # 公网隧道（frpc）。相关脚本与配置未纳入版本库，缺失时会明确报错。
    [switch]$WithTunnel,

    # 跳过镜像构建，直接用现有镜像启动。只改了配置没改代码时能省一两分钟。
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $PSScriptRoot "runtime"
$FrpRoot = Join-Path $RuntimeRoot "frp"
$PidRoot = Join-Path $RuntimeRoot "pids"
$ModelCacheRoot = Join-Path $RuntimeRoot "model-cache"
$PackageCacheRoot = Join-Path $RuntimeRoot "package-cache"
$TemporaryRoot = Join-Path $RuntimeRoot "temp"
$ComposeFile = Join-Path $PSScriptRoot "docker\compose.yaml"
$DockerEnvFile = Join-Path $PSScriptRoot "docker\.env"
$RootEnvFile = Join-Path $ProjectRoot ".env"

# ── 工具函数 ────────────────────────────────────────────────────────────────────

function Test-ListeningPort {
    param([int]$Port)

    return $null -ne (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1)
}

function Wait-Http {
    param([string]$Url, [string]$Name, [int]$Attempts = 30)

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 | Out-Null
            Write-Host "[$Name] ready: $Url"
            return
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    throw "[$Name] did not become ready: $Url"
}

function Start-ManagedProcess {
    param([string]$Name, [string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory)

    $pidFile = Join-Path $PidRoot "$Name.pid"
    if (Test-Path -LiteralPath $pidFile) {
        $previousPid = [int](Get-Content -LiteralPath $pidFile -Raw)
        if (Get-Process -Id $previousPid -ErrorAction SilentlyContinue) {
            Write-Host "[$Name] already started by this script (pid=$previousPid)"
            return
        }
        Remove-Item -LiteralPath $pidFile -Force
    }

    try {
        $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments `
            -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru
    } catch {
        throw "[$Name] failed to start: $($_.Exception.Message)"
    }
    Set-Content -LiteralPath $pidFile -Value $process.Id -NoNewline
    Write-Host "[$Name] started (pid=$($process.Id))"
}

function Find-FirstExistingPath {
    param([string[]]$Candidates, [string]$Description)

    $match = $Candidates |
        Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
        Select-Object -First 1
    if (-not $match) {
        throw "Unable to find $Description. Pass its explicit path to this script."
    }
    return (Get-Item -LiteralPath $match).FullName
}

function Read-EnvFile {
    param([string]$Path)

    $values = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $values
    }
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*(#|$)') { continue }
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $values[$matches[1]] = $matches[2].Trim('"')
        }
    }
    return $values
}

function New-RandomSecret {
    param([int]$Length = 32)

    # 只取字母数字：口令要拼进 redis:// 和 mysql+aiomysql:// 这类 DSN，一旦含
    # @ : / # 等字符就必须百分号转义，而转义漏一处的表现是「连不上」而非「报错」。
    # 去掉了 l/I/O/0/1 这些易混字符，方便人工核对。
    $alphabet = 'abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    $bytes = [byte[]]::new($Length)
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    return -join ($bytes | ForEach-Object { $alphabet[$_ % $alphabet.Length] })
}

function Write-EnvFileUtf8 {
    param([string]$Path, [string[]]$Lines)

    # 必须是不带 BOM 的 UTF-8：Set-Content -Encoding UTF8 在 Windows PowerShell 5.1
    # 下会写入 BOM，而 compose 解析 .env 时会把它算进第一个键名里。
    [System.IO.File]::WriteAllLines($Path, $Lines, (New-Object System.Text.UTF8Encoding $false))
}

function Initialize-DockerEnvFile {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        # 已有的口令绝不改动：MySQL 口令必须与数据卷里已初始化的那份保持一致，
        # 重新生成会让数据库直接连不上。轮换步骤见 README「凭据轮换」。
        $existing = Read-EnvFile -Path $Path
        $missing = @('MYSQL_PASSWORD', 'MYSQL_ROOT_PASSWORD', 'REDIS_PASSWORD', 'QDRANT_API_KEY') |
            Where-Object { -not $existing[$_] }
        if ($missing) {
            throw "deployment/docker/.env 缺少口令项: $($missing -join ', ')。补齐后重跑，或删除该文件让脚本重新生成（会丢失 MySQL 数据卷的访问权限）。"
        }

        # HEALTH_DETAIL_TOKEN 是后加的项，早先生成的 .env 里没有。它和数据卷无关，
        # 补一行即刻生效，所以就地追加而不是像上面那样要求人工处理。
        if (-not $existing['HEALTH_DETAIL_TOKEN']) {
            $lines = New-Object System.Collections.Generic.List[string]
            $lines.AddRange([string[]][System.IO.File]::ReadAllLines($Path))
            $lines.Add("")
            $lines.Add("# /health 明细令牌（新增项，由 start-all.ps1 补写）。说明见 .env.example。")
            $lines.Add("HEALTH_DETAIL_TOKEN=$(New-RandomSecret 40)")
            Write-EnvFileUtf8 -Path $Path -Lines $lines.ToArray()
            Write-Host "[secrets] 已为 deployment/docker/.env 补写 HEALTH_DETAIL_TOKEN"
        }
        return
    }

    $content = @(
        "# 由 deployment/start-all.ps1 自动生成，不入版本库。字段说明见 .env.example。",
        "WEB_PORT=80",
        "WORKER_REPLICAS=2",
        "",
        "MYSQL_PASSWORD=$(New-RandomSecret 32)",
        "MYSQL_ROOT_PASSWORD=$(New-RandomSecret 32)",
        "REDIS_PASSWORD=$(New-RandomSecret 32)",
        "QDRANT_API_KEY=$(New-RandomSecret 40)",
        "",
        "# /health 的 dependencies[].detail 默认不对外返回，带这个令牌的请求才看得到。",
        "HEALTH_DETAIL_TOKEN=$(New-RandomSecret 40)",
        "",
        "# 宿主机侧的调试端口，均只绑 127.0.0.1",
        "MYSQL_PORT=3307",
        "REDIS_PORT=6380",
        "QDRANT_PORT=6334"
    )
    Write-EnvFileUtf8 -Path $Path -Lines $content
    Write-Host "[secrets] 已生成 deployment/docker/.env（4 个随机口令 + 1 个明细令牌）"
}

function Assert-RootEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "缺少项目根目录的 .env。先执行 copy .env.example .env 并填入 LLM_API_KEY。"
    }
    $values = Read-EnvFile -Path $Path
    if (-not $values['LLM_API_KEY'] -or $values['LLM_API_KEY'] -match 'your_|REPLACE_|change-me') {
        throw "根目录 .env 里的 LLM_API_KEY 尚未填写。每个流程节点都要调 LLM，缺它任务必然失败。"
    }
    if ($values['USE_TAVILY'] -match '^(?i:true|1|yes)$' -and -not $values['TAVILY_API_KEY']) {
        throw "USE_TAVILY=true 但 TAVILY_API_KEY 为空。改用免费搜索请设 USE_TAVILY=false。"
    }
}

# ── 准备 ────────────────────────────────────────────────────────────────────────

Assert-RootEnv -Path $RootEnvFile
Initialize-DockerEnvFile -Path $DockerEnvFile

$dockerEnv = Read-EnvFile -Path $DockerEnvFile
$webPort = if ($dockerEnv['WEB_PORT']) { [int]$dockerEnv['WEB_PORT'] } else { 80 }

$docker = (Get-Command docker -ErrorAction SilentlyContinue)
if (-not $docker) {
    throw "未找到 docker 命令。请先安装并启动 Docker Desktop。"
}
# 不要写成 `docker info 2>&1`：Windows PowerShell 5.1 会把原生命令的 stderr 包成
# ErrorRecord，在 $ErrorActionPreference='Stop' 下即使退出码为 0 也会抛异常。
& $docker.Source info --format "{{.ServerVersion}}" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker 守护进程未响应。请先启动 Docker Desktop 再重跑。"
}

New-Item -ItemType Directory -Path $PidRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $ModelCacheRoot "huggingface") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $ModelCacheRoot "torch") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PackageCacheRoot "pip") -Force | Out-Null
New-Item -ItemType Directory -Path $TemporaryRoot -Force | Out-Null

# 模型权重和 PyTorch 扩展只能存放在项目运行目录，避免持续占用系统盘
$env:HF_HOME = [System.IO.Path]::Combine($ModelCacheRoot, "huggingface")
$env:HF_HUB_CACHE = [System.IO.Path]::Combine($env:HF_HOME, "hub")
$env:TRANSFORMERS_CACHE = $env:HF_HUB_CACHE
$env:TORCH_HOME = [System.IO.Path]::Combine($ModelCacheRoot, "torch")
$env:PIP_CACHE_DIR = [System.IO.Path]::Combine($PackageCacheRoot, "pip")
$env:TEMP = $TemporaryRoot
$env:TMP = $TemporaryRoot

$systemPython = Get-Command python -ErrorAction SilentlyContinue
$python = Find-FirstExistingPath -Description "a project Python executable" -Candidates @(
    $PythonExe,
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    $(if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" }),
    # 排除 WindowsApps 下那个 python.exe —— 它是微软商店的占位符，文件确实存在，
    # 执行时却只弹商店安装页。不排掉就会顶掉后面真实的解释器。
    $(if ($systemPython -and $systemPython.Source -notmatch '\\WindowsApps\\') { $systemPython.Source })
)
& $python -c "import sys" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "选中的 Python 无法执行: $python。用 -PythonExe 指定项目环境里的 python.exe。"
}
& $python -c "import torch, sentence_transformers" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Python 环境缺少模型服务依赖: $python。在该环境执行 pip install -r requirements-model.txt，或用 -PythonExe 指定正确的环境。"
}

# ── 1. 模型服务（宿主机）────────────────────────────────────────────────────────
# 先拉起来再去构建镜像：冷启动要顺序装载 embed(2.2GB) + rerank(2.2GB) + translate
# 三个模型、全部预热完才开始监听，本机实测约 100 秒，正好和镜像构建重叠掉。

if (-not (Test-ListeningPort 8100)) {
    if ($ModelServerHost -eq "0.0.0.0") {
        Write-Host "[model-server] 监听 0.0.0.0:8100（容器访问所必需）——该端口无鉴权，请确认防火墙已限制入站"
    }
    Start-ManagedProcess -Name "model-server" -FilePath $python `
        -Arguments @("-m", "uvicorn", "app.model_server:app", "--host", $ModelServerHost, "--port", "8100") `
        -WorkingDirectory $ProjectRoot
}

# ── 2. 容器栈 ───────────────────────────────────────────────────────────────────

$env:WORKER_REPLICAS = $WorkerCount     # 环境变量优先级高于 .env，覆盖其中的默认值
$composeArgs = @("compose", "-f", $ComposeFile, "up", "-d")
if (-not $SkipBuild) { $composeArgs += "--build" }

Write-Host "[docker] 启动容器栈（worker × $WorkerCount）"
# docker compose 的构建与编排进度全部走 stderr。在 $ErrorActionPreference='Stop' 下，
# 只要调用方把脚本输出重定向过（`.\start-all.ps1 *> log.txt`），Windows PowerShell 5.1
# 就会把这些进度行包成 ErrorRecord 并当成终止错误抛出——明明构建是成功的。
# 这里临时放行，成败一律以 $LASTEXITCODE 为准。
$previousActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $docker.Source @composeArgs
$ErrorActionPreference = $previousActionPreference
if ($LASTEXITCODE -ne 0) {
    throw "容器栈启动失败。用 docker compose -f deployment/docker/compose.yaml logs 查看详情。"
}

# ── 3. 等待就绪 ─────────────────────────────────────────────────────────────────

Wait-Http -Url "http://127.0.0.1:8100/health" -Name "model-server" -Attempts 180
Wait-Http -Url "http://127.0.0.1:$webPort/health" -Name "web" -Attempts 60

# ── 4. 依赖校验 ─────────────────────────────────────────────────────────────────
# /health 恒返回 200，健康度在 body 里。刚启动的栈本就该全绿；降级是运行期的容错
# 能力，不是一次成功部署的可接受终态，所以这里任何一项失败都判定为启动失败。

# /health 默认不返回 detail（该端点公网可达）。带上令牌才拿得到下面要打印的原因，
# 没配令牌也不影响判定——status 和 failed 本来就是公开的。
$healthHeaders = @{}
if ($dockerEnv['HEALTH_DETAIL_TOKEN']) {
    $healthHeaders['X-Health-Token'] = $dockerEnv['HEALTH_DETAIL_TOKEN']
}

# 不用 Invoke-RestMethod：FastAPI 的 Content-Type 不带 charset，Windows PowerShell 5.1
# 遇到这种情况按 ISO-8859-1 解码，detail 里的中文会变成乱码。自己按 UTF-8 解字节。
$raw = Invoke-WebRequest -Uri "http://127.0.0.1:$webPort/health" -Headers $healthHeaders -UseBasicParsing -TimeoutSec 15
$health = [System.Text.Encoding]::UTF8.GetString($raw.RawContentStream.ToArray()) | ConvertFrom-Json
Write-Host ""
Write-Host "依赖校验："
foreach ($dep in $health.dependencies) {
    $mark = if ($dep.skipped) { "跳过" } elseif ($dep.ok) { "正常" } else { "失败" }
    Write-Host ("  {0,-14} {1}  {2}" -f $dep.name, $mark, $dep.detail)
}
Write-Host ""

if ($health.status -ne "ok") {
    throw "依赖未全部就绪: $($health.failed -join ', ')。上面每一行的 detail 即为原因；模型服务一项失败时先确认它监听在 0.0.0.0 而非 127.0.0.1。"
}

# ── 5. 公网隧道（可选）──────────────────────────────────────────────────────────

if ($WithTunnel) {
    $tunnelScript = Join-Path $PSScriptRoot "start-public-tunnel.ps1"
    if (-not (Test-Path -LiteralPath $tunnelScript)) {
        throw "缺少 deployment/start-public-tunnel.ps1。隧道脚本与 frpc 配置含服务器凭据，未纳入版本库，需在本机单独准备。"
    }
    & $tunnelScript
}

$entry = if ($webPort -eq 80) { "http://127.0.0.1/" } else { "http://127.0.0.1:$webPort/" }
Write-Host "Deep Research Agent 已启动。本机入口: $entry"
