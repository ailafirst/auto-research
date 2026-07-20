[CmdletBinding()]
param(
    [ValidateRange(0, 8)]
    [int]$WorkerCount = 2,
    [switch]$SkipFrontendBuild,
    [switch]$SkipTunnel,
    [string]$PythonExe = "",
    [string]$ArqExe = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $PSScriptRoot "runtime"
$NginxRoot = Join-Path $RuntimeRoot "nginx"
$FrpRoot = Join-Path $RuntimeRoot "frp"
$PidRoot = Join-Path $RuntimeRoot "pids"
$ModelCacheRoot = Join-Path $RuntimeRoot "model-cache"
$PackageCacheRoot = Join-Path $RuntimeRoot "package-cache"
$TemporaryRoot = Join-Path $RuntimeRoot "temp"
$WebRoot = Join-Path $ProjectRoot "web"

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
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

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
        if ($Name -eq "frpc") {
            throw "[frpc] Windows refused to create the approved client process: $($_.Exception.Message). Integrity verification already passed; local services are running. Obtain endpoint-security or permission approval for this file through the normal process, then rerun to start the public tunnel."
        }
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

foreach ($path in @($RuntimeRoot, $NginxRoot, $FrpRoot)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing deployment runtime path: $path"
    }
}
New-Item -ItemType Directory -Path $PidRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $ModelCacheRoot "huggingface") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $ModelCacheRoot "torch") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PackageCacheRoot "pip") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $PackageCacheRoot "npm") -Force | Out-Null
New-Item -ItemType Directory -Path $TemporaryRoot -Force | Out-Null

# 模型权重和 PyTorch 扩展只能存放在项目运行目录，避免持续占用系统盘。
$env:HF_HOME = [System.IO.Path]::Combine($ModelCacheRoot, "huggingface")
$env:HF_HUB_CACHE = [System.IO.Path]::Combine($env:HF_HOME, "hub")
$env:TRANSFORMERS_CACHE = $env:HF_HUB_CACHE
$env:TORCH_HOME = [System.IO.Path]::Combine($ModelCacheRoot, "torch")
$env:PIP_CACHE_DIR = [System.IO.Path]::Combine($PackageCacheRoot, "pip")
$env:npm_config_cache = [System.IO.Path]::Combine($PackageCacheRoot, "npm")
$env:TEMP = $TemporaryRoot
$env:TMP = $TemporaryRoot

$systemPython = Get-Command python -ErrorAction SilentlyContinue
$python = Find-FirstExistingPath -Description "a project Python executable" -Candidates @(
    $PythonExe,
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    $(if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" }),
    $(if ($systemPython) { $systemPython.Source })
)
$pythonDirectory = Split-Path -Parent $python
$environmentRoot = if ((Split-Path -Leaf $pythonDirectory) -eq "Scripts") {
    Split-Path -Parent $pythonDirectory
} else {
    $pythonDirectory
}
& $python -c "import arq" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "The selected Python environment does not contain arq. Install requirements.txt in that environment."
}
$systemArq = Get-Command arq -ErrorAction SilentlyContinue
$arq = @(
    $ArqExe,
    (Join-Path $environmentRoot "Scripts\arq.exe"),
    $(if ($systemArq) { $systemArq.Source })
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1

if (-not (Test-ListeningPort 6379)) {
    $docker = (Get-Command docker -ErrorAction Stop).Source
    Write-Host "[redis] starting with Docker Compose"
    & $docker compose -f (Join-Path $ProjectRoot "docker-compose.yml") up -d redis
    if ($LASTEXITCODE -ne 0) {
        throw "Redis failed to start."
    }
}
if (-not (Test-ListeningPort 6379)) {
    for ($attempt = 1; $attempt -le 20 -and -not (Test-ListeningPort 6379); $attempt++) {
        Start-Sleep -Seconds 1
    }
}
if (-not (Test-ListeningPort 6379)) {
    throw "Redis is not listening on 127.0.0.1:6379."
}
Write-Host "[redis] ready"

if (-not $SkipFrontendBuild) {
    $npm = (Get-Command npm -ErrorAction Stop).Source
    Push-Location $WebRoot
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $WebRoot "node_modules"))) {
            Write-Host "[web] installing locked dependencies"
            & $npm ci
            if ($LASTEXITCODE -ne 0) {
                throw "Frontend dependency installation failed."
            }
        }
        Write-Host "[web] building production assets"
        & $npm run build
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build failed."
        }
    } finally {
        Pop-Location
    }
}

if (-not (Test-ListeningPort 8100)) {
    Start-ManagedProcess -Name "model-server" -FilePath $python `
        -Arguments @("-m", "uvicorn", "app.model_server:app", "--host", "127.0.0.1", "--port", "8100") `
        -WorkingDirectory $ProjectRoot
}
Wait-Http -Url "http://127.0.0.1:8100/health" -Name "model-server"

if (-not (Test-ListeningPort 8000)) {
    Start-ManagedProcess -Name "api" -FilePath $python `
        -Arguments @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $ProjectRoot
}
Wait-Http -Url "http://127.0.0.1:8000/health" -Name "api"

for ($index = 1; $index -le $WorkerCount; $index++) {
    if ($arq) {
        Start-ManagedProcess -Name "worker-$index" -FilePath $arq `
            -Arguments @("app.worker.WorkerSettings") -WorkingDirectory $ProjectRoot
    } else {
        # arq 的 Python 模块入口与已选 Python 环境绑定，避免依赖 Scripts/arq.exe 或 PATH。
        Start-ManagedProcess -Name "worker-$index" -FilePath $python `
            -Arguments @("-m", "arq", "app.worker.WorkerSettings") -WorkingDirectory $ProjectRoot
    }
}

if (-not (Test-ListeningPort 80)) {
    Start-ManagedProcess -Name "nginx" -FilePath (Join-Path $NginxRoot "nginx.exe") `
        -Arguments @("-p", $NginxRoot) -WorkingDirectory $NginxRoot
    for ($attempt = 1; $attempt -le 10 -and -not (Test-ListeningPort 80); $attempt++) {
        Start-Sleep -Seconds 1
    }
}
if (-not (Test-ListeningPort 80)) {
    throw "Nginx is not listening on port 80. Run deployment/remove-legacy-c-drive.ps1 as Administrator if an old C: Nginx still owns the port."
}
Wait-Http -Url "http://127.0.0.1/" -Name "nginx"

if (-not $SkipTunnel) {
    Start-ManagedProcess -Name "frpc" -FilePath (Join-Path $FrpRoot "frpc.exe") `
        -Arguments @("-c", (Join-Path $FrpRoot "frpc.toml")) -WorkingDirectory $FrpRoot
    Start-Sleep -Seconds 2
    $frpcLog = Join-Path $FrpRoot "frpc.log"
    if (Test-Path -LiteralPath $frpcLog) {
        $tail = Get-Content -LiteralPath $frpcLog -Tail 10
        if ($tail -match "start proxy success") {
            Write-Host "[frpc] tunnel connected"
        } else {
            Write-Warning "[frpc] process started but connection success was not found in frpc.log yet."
        }
    }
}

Write-Host "Deep Research Agent started. Local entry: http://127.0.0.1/"
