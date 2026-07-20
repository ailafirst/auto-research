[CmdletBinding()]
param(
    [ValidateRange(0, 8)]
    [int]$WorkerCount = 2,
    [switch]$SkipFrontendBuild,
    [switch]$SkipTunnel,
    [switch]$PrepareOnly,
    [string]$PythonExe = "",
    [string]$ArqExe = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $PSScriptRoot "runtime"
$TemplateRoot = Join-Path $PSScriptRoot "templates"
$SecretsFile = Join-Path $RuntimeRoot "secrets.env"
$RootEnvFile = Join-Path $ProjectRoot ".env"
$FrpRoot = Join-Path $RuntimeRoot "frp"
$FrpConfig = Join-Path $FrpRoot "frpc.toml"
$FrpVerifier = Join-Path $PSScriptRoot "verify-frpc.ps1"
$NginxRoot = Join-Path $RuntimeRoot "nginx"
$NginxConfig = Join-Path $NginxRoot "conf\nginx.conf"

function Ensure-PrivateFile {
    param([string]$Target, [string]$Template)

    if (-not (Test-Path -LiteralPath $Target)) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
        Copy-Item -LiteralPath $Template -Destination $Target
        return $false
    }
    return $true
}

function Test-PlaceholderFree {
    param([string]$Path)

    $content = Get-Content -LiteralPath $Path -Raw
    return $content -notmatch "REPLACE_[A-Z0-9_]+|your_[a-z0-9_]+_here"
}

function Test-RequiredEnvironment {
    param([string]$Path)

    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match "^\s*(#|$)") {
            continue
        }
        if ($line -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$") {
            $values[$matches[1]] = $matches[2].Trim('"')
        }
    }

    $required = @("LLM_API_KEY", "SECRET_KEY")
    if ($values["USE_TAVILY"] -match "^(?i:true|1|yes)$") {
        $required += "TAVILY_API_KEY"
    }
    foreach ($name in $required) {
        $value = $values[$name]
        if (-not $value -or $value -match "REPLACE_[A-Z0-9_]+|your_[a-z0-9_]+_here|change-me-in-production") {
            return $false
        }
    }
    return $true
}

function Import-PrivateEnvironment {
    param([string]$Path)

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match "^\s*(#|$)") {
            continue
        }
        if ($line -notmatch "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$") {
            throw "Invalid environment-variable syntax in private settings file."
        }
        $name = $matches[1]
        $value = $matches[2]
        if ($value.Length -ge 2 -and $value.StartsWith('"') -and $value.EndsWith('"')) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        Set-Item -Path "Env:$name" -Value $value
    }
}

New-Item -ItemType Directory -Path $RuntimeRoot, $FrpRoot, $NginxRoot -Force | Out-Null

# 根目录 .env 是既有应用配置的标准位置；runtime/secrets.env 用于希望把部署私有
# 配置与应用配置分离的场景。两者均不存在时才创建无敏感模板。
$secretsSource = $null
$secretsReady = $false
if ((Test-Path -LiteralPath $SecretsFile) -and (Test-PlaceholderFree -Path $SecretsFile)) {
    $secretsSource = $SecretsFile
    $secretsReady = $true
} elseif (Test-Path -LiteralPath $RootEnvFile) {
    $secretsSource = $RootEnvFile
    $secretsReady = $true
} elseif (Test-Path -LiteralPath $SecretsFile) {
    $secretsSource = $SecretsFile
} else {
    [void](Ensure-PrivateFile -Target $SecretsFile -Template (Join-Path $TemplateRoot "secrets.env.example"))
    $secretsSource = $SecretsFile
}
$frpReady = $SkipTunnel -or (Ensure-PrivateFile -Target $FrpConfig -Template (Join-Path $TemplateRoot "frpc.toml.example"))

if (-not $PrepareOnly) {
    if (-not $secretsReady -or -not (Test-RequiredEnvironment -Path $secretsSource)) {
        throw "Private settings are missing or incomplete. Fill root .env or deployment/runtime/secrets.env locally, then rerun."
    }
    if (-not $SkipTunnel -and (-not $frpReady -or -not (Test-PlaceholderFree -Path $FrpConfig))) {
        throw "Private tunnel template created or incomplete: deployment/runtime/frp/frpc.toml. Fill it locally, then rerun."
    }
}

if (-not (Test-Path -LiteralPath $NginxConfig)) {
    $template = Get-Content -LiteralPath (Join-Path $TemplateRoot "nginx.conf.template") -Raw
    $portableRoot = $ProjectRoot.Replace("\", "/")
    New-Item -ItemType Directory -Path (Split-Path -Parent $NginxConfig) -Force | Out-Null
    Set-Content -LiteralPath $NginxConfig -Value $template.Replace("__PROJECT_ROOT__", $portableRoot) -NoNewline
}
foreach ($path in @("logs", "temp\client_body_temp", "temp\fastcgi_temp", "temp\proxy_temp", "temp\scgi_temp", "temp\uwsgi_temp")) {
    New-Item -ItemType Directory -Path (Join-Path $NginxRoot $path) -Force | Out-Null
}

if ($PrepareOnly) {
    Write-Host "Private runtime templates and Nginx configuration are prepared."
    return
}

if (-not (Test-Path -LiteralPath (Join-Path $NginxRoot "nginx.exe"))) {
    throw "Nginx is missing from deployment/runtime/nginx. Install a trusted Nginx Windows package there before starting."
}
if (-not $SkipTunnel -and -not (Test-Path -LiteralPath (Join-Path $FrpRoot "frpc.exe"))) {
    throw "frpc.exe is missing from deployment/runtime/frp. Install a trusted, approved frp client there before starting."
}
if (-not $SkipTunnel) {
    & $FrpVerifier
}

Import-PrivateEnvironment -Path $secretsSource

& (Join-Path $PSScriptRoot "start-all.ps1") -WorkerCount $WorkerCount `
    -SkipFrontendBuild:$SkipFrontendBuild -SkipTunnel:$SkipTunnel `
    -PythonExe $PythonExe -ArqExe $ArqExe
