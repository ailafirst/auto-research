#Requires -RunAsAdministrator
<#
.SYNOPSIS
    删除早期装在 C 盘的 Nginx / frp 遗留目录。

.DESCRIPTION
    一次性清理。这两份东西现在都有了替代品：

        C:\nginx-1.28.0  →  容器里的 nginx（deployment/docker/compose.yaml）
        C:\frp           →  deployment/runtime/frp/

    动作是 Remove-Item -Recurse -Force，没有回头路，所以每个目标删除前都要先确认
    它的替代品确实在位，确认不了就拒绝执行。

    判据取「nginx 容器已被创建过」而不是「正在运行」，是因为后者会死锁：旧 Nginx
    占着 80 端口时，容器 nginx 恰恰起不来，而清掉旧 Nginx 正是本脚本要做的事。
    容器只要被 compose 创建过（哪怕因端口冲突启动失败）就足以证明替代品已经就位。

    先用 -WhatIf 看它打算做什么：

        .\deployment\remove-legacy-c-drive.ps1 -WhatIf

.EXAMPLE
    .\deployment\remove-legacy-c-drive.ps1

.EXAMPLE
    # 无人值守，跳过确认提示
    .\deployment\remove-legacy-c-drive.ps1 -Confirm:$false
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = "High")]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RuntimeRoot = Join-Path $PSScriptRoot "runtime"
$ComposeFile = Join-Path $PSScriptRoot "docker\compose.yaml"

function Test-ContainerNginxDeployed {
    if (-not (Test-Path -LiteralPath $ComposeFile)) {
        return $false
    }
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $docker) {
        return $false
    }

    # compose 把进度写到 stderr；Windows PowerShell 5.1 在 $ErrorActionPreference='Stop'
    # 下会把它包成 ErrorRecord 抛出，这里临时放行，成败以 $LASTEXITCODE 为准。
    $previousActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $services = & $docker.Source compose -f $ComposeFile ps -a --services
    $ErrorActionPreference = $previousActionPreference

    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    # 工程从未部署过时返回空，正是要拒绝的情况
    return ($services -contains "nginx")
}

$targets = @(
    @{
        Name        = "Nginx"
        Path        = "C:\nginx-1.28.0"
        Replacement = "容器 nginx —— 先跑一次 deployment\start-all.ps1"
        IsReady     = { Test-ContainerNginxDeployed }
    },
    @{
        Name        = "frp 客户端"
        Path        = "C:\frp"
        Replacement = "deployment\runtime\frp\frpc.exe"
        IsReady     = { Test-Path -LiteralPath (Join-Path $RuntimeRoot "frp\frpc.exe") }
    }
)

$removed = 0
foreach ($target in $targets) {
    if (-not (Test-Path -LiteralPath $target.Path)) {
        Write-Host "[$($target.Name)] 遗留目录不存在，跳过: $($target.Path)"
        continue
    }
    if (-not (& $target.IsReady)) {
        throw "拒绝删除 $($target.Path)：替代品未就位（$($target.Replacement)）。"
    }

    # 先停掉从该目录启动的进程，否则文件被占用会删不干净。
    # 匹配串带上路径分隔符，避免 C:\frp 顺手匹配到 C:\frp-backup 之类的邻居目录。
    # 用 Get-Process 而非 Get-CimInstance：后者会延迟加载 CimCmdlets 模块，而模块内部
    # 那批 Set-Alias 在 -WhatIf 模式下会被逐条打印出来，刷十几行与本脚本无关的噪音。
    $legacyProcesses = Get-Process | Where-Object { $_.Path -like "$($target.Path)\*" }
    foreach ($process in $legacyProcesses) {
        if ($PSCmdlet.ShouldProcess("$($process.ProcessName) (pid=$($process.Id))", "停止遗留进程")) {
            Stop-Process -Id $process.Id -Force
            Write-Host "[$($target.Name)] 已停止遗留进程 $($process.ProcessName) (pid=$($process.Id))"
        }
    }

    if ($PSCmdlet.ShouldProcess($target.Path, "递归删除")) {
        Remove-Item -LiteralPath $target.Path -Recurse -Force
        Write-Host "[$($target.Name)] 已删除: $($target.Path)"
        $removed++
    }
}

if ($removed -eq 0) {
    Write-Host "没有需要清理的 C 盘遗留目录。"
} else {
    Write-Host "C 盘遗留文件清理完成。项目请用 deployment\start-all.ps1 启动。"
}
