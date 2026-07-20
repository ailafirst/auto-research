[CmdletBinding()]
param(
    [ValidateRange(0, 8)]
    [int]$WorkerCount = 2,
    [switch]$SkipFrontendBuild,
    [switch]$PrepareOnly,
    [string]$PythonExe = "",
    [string]$ArqExe = ""
)

$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "bootstrap-public.ps1") `
    -WorkerCount $WorkerCount `
    -SkipFrontendBuild:$SkipFrontendBuild `
    -SkipTunnel `
    -PrepareOnly:$PrepareOnly `
    -PythonExe $PythonExe `
    -ArqExe $ArqExe
