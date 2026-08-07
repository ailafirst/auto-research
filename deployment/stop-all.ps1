<#
.SYNOPSIS
    停止 Deep Research Agent 的全部服务。

.DESCRIPTION
    默认只停不删：容器保留、数据卷保留（MySQL 里的任务记录和 Qdrant 里的向量都还在），
    下次 start-all.ps1 可以直接起回来。

    要连容器一起删用 -RemoveContainers；数据卷不在本脚本的处理范围内，确实要清空数据时
    请显式执行 `docker compose -f deployment/docker/compose.yaml down -v`。

.EXAMPLE
    .\deployment\stop-all.ps1

.EXAMPLE
    # 只重启容器栈，模型服务留着（冷启动要 100 秒，不值得反复重来）
    .\deployment\stop-all.ps1 -KeepModelServer
#>
[CmdletBinding()]
param(
    [switch]$KeepModelServer,
    [switch]$RemoveContainers
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PidRoot = Join-Path $PSScriptRoot "runtime\pids"
$ComposeFile = Join-Path $PSScriptRoot "docker\compose.yaml"

function Stop-RecordedProcess {
    param([string]$Name)

    $pidFile = Join-Path $PidRoot "$Name.pid"
    if (-not (Test-Path -LiteralPath $pidFile)) {
        return
    }
    $servicePid = [int](Get-Content -LiteralPath $pidFile -Raw)
    if (Get-Process -Id $servicePid -ErrorAction SilentlyContinue) {
        Stop-Process -Id $servicePid -Force
        Write-Host "[$Name] stopped (pid=$servicePid)"
    }
    Remove-Item -LiteralPath $pidFile -Force
}

Stop-RecordedProcess -Name "frpc"

if (-not $KeepModelServer) {
    Stop-RecordedProcess -Name "model-server"
    # pid 文件可能因为异常退出而丢失，按端口兜底一次
    $owners = Get-NetTCPConnection -State Listen -LocalPort 8100 -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($owner in $owners) {
        $process = Get-Process -Id $owner -ErrorAction SilentlyContinue
        if ($process -and $process.ProcessName -eq "python") {
            Stop-Process -Id $owner -Force
            Write-Host "[model-server] stopped by port (pid=$owner)"
        }
    }
}

$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    Write-Warning "未找到 docker 命令，容器栈未停止。"
} else {
    $action = if ($RemoveContainers) { "down" } else { "stop" }
    # 同 start-all.ps1：compose 的进度走 stderr，输出被重定向时会在 Stop 模式下
    # 被当成终止错误。成败以 $LASTEXITCODE 为准。
    $previousActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $docker.Source compose -f $ComposeFile $action
    $ErrorActionPreference = $previousActionPreference
    if ($LASTEXITCODE -ne 0) {
        throw "容器栈未能干净停止（docker compose $action 返回 $LASTEXITCODE）。"
    }
    Write-Host "[docker] 容器栈已 $action"
}

Write-Host "Deep Research Agent 已停止。"
