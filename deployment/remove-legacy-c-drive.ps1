#Requires -RunAsAdministrator
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $PSScriptRoot "runtime"
$LegacyNginx = "C:\nginx-1.28.0"
$LegacyFrp = "C:\frp"

foreach ($required in @(
    (Join-Path $RuntimeRoot "nginx\nginx.exe"),
    (Join-Path $RuntimeRoot "nginx\conf\nginx.conf"),
    (Join-Path $RuntimeRoot "frp\frpc.exe"),
    (Join-Path $RuntimeRoot "frp\frpc.toml")
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Migration is incomplete; refusing to remove legacy path. Missing: $required"
    }
}

$legacyProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.ExecutablePath -like "$LegacyNginx*" -or $_.ExecutablePath -like "$LegacyFrp*"
}
foreach ($process in $legacyProcesses) {
    Stop-Process -Id $process.ProcessId -Force
    Write-Host "Stopped legacy process $($process.Name) (pid=$($process.ProcessId))"
}

foreach ($path in @($LegacyFrp, $LegacyNginx)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
        Write-Host "Removed legacy path: $path"
    }
}

Write-Host "Legacy C: deployment files removed. Start the project with deployment\start-all.ps1."
