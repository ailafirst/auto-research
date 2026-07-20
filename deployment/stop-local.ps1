[CmdletBinding()]
param(
    [switch]$KeepRedis
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PidRoot = Join-Path $PSScriptRoot "runtime\pids"

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

function Stop-ProjectPort {
    param([int]$Port)

    $connections = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    $processIds = @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($processId in $processIds) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process -and $process.ProcessName -eq "python") {
            Stop-Process -Id $processId -Force
            Write-Host ("[port-{0}] project Python service stopped (pid={1})" -f $Port, $processId)
        }
    }
}

$workerPidFiles = Get-ChildItem -LiteralPath $PidRoot -Filter "worker-*.pid" -ErrorAction SilentlyContinue
foreach ($workerPidFile in $workerPidFiles) {
    Stop-RecordedProcess -Name $workerPidFile.BaseName
}
Stop-RecordedProcess -Name "api"
Stop-RecordedProcess -Name "model-server"
Stop-RecordedProcess -Name "nginx"
Stop-ProjectPort -Port 8000
Stop-ProjectPort -Port 8100

if (-not $KeepRedis) {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($docker) {
        & $docker.Source compose -f (Join-Path $ProjectRoot "docker-compose.yml") stop redis
        if ($LASTEXITCODE -ne 0) {
            throw "Redis container did not stop cleanly."
        }
        Write-Host "[redis] stopped"
    } else {
        Write-Warning "[redis] Docker was not found; Redis was not stopped."
    }
}

Write-Host "Local application services stopped."
