<#
.SYNOPSIS
    One-command setup for the academic-automl-platform Docker stack.
.EXAMPLE
    .\run.ps1              # build + start full stack, wait until healthy
    .\run.ps1 -Test        # run the full test suite in an ephemeral container first
    .\run.ps1 -SkipBuild   # start using existing images
    .\run.ps1 -Down        # stop the stack
#>
[CmdletBinding()]
param(
    [switch]$Test,
    [switch]$Down,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Write-Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Fail([string]$Message) { Write-Host "ERROR: $Message" -ForegroundColor Red }

if ($Down) {
    Write-Step "Stopping stack"
    docker compose down
    exit $LASTEXITCODE
}

Write-Step "Checking environment file"
if (-not (Test-Path ".env")) {
    $password = [System.Guid]::NewGuid().ToString("N").Substring(0, 24)
    $content = (Get-Content ".env.example") -replace "^REDIS_PASSWORD=.*$", "REDIS_PASSWORD=$password"
    [System.IO.File]::WriteAllLines((Join-Path (Get-Location) ".env"), $content)
    Write-Host "    Created .env from .env.example with a generated REDIS_PASSWORD."
} else {
    Write-Host "    Existing .env left untouched."
}

Write-Step "Validating Docker engine"
$null = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Docker daemon is not reachable."
    Write-Host "       Start Docker Desktop and retry. Install guide: https://docs.docker.com/get-docker/" -ForegroundColor Yellow
    exit 2
}
$null = docker compose version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Docker Compose v2 plugin not found."
    exit 2
}
Write-Host "    Docker engine and Compose plugin OK."

if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}

if ($Test) {
    Write-Step "Building test image"
    docker compose --profile test build tests
    if ($LASTEXITCODE -ne 0) { Write-Fail "Test image build failed."; exit 3 }

    Write-Step "Running full test suite inside container"
    docker compose --profile test run --rm tests
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Containerized test suite failed; aborting deployment."
        exit 3
    }
}

Write-Step "Building and starting stack (api, worker, beat, redis)"
$upArgs = @("compose", "up", "-d")
if (-not $SkipBuild) { $upArgs += "--build" }
& docker @upArgs
if ($LASTEXITCODE -ne 0) { Write-Fail "docker compose up failed."; exit 2 }

Write-Step "Waiting for API health at http://localhost:8000/api/v1/health"
$deadline = (Get-Date).AddSeconds(120)
$healthy = $false
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) { $healthy = $true; break }
    } catch {
        # not ready yet
    }
    Write-Host "." -NoNewline
    Start-Sleep -Seconds 2
}
Write-Host ""
if (-not $healthy) {
    Write-Fail "API did not become healthy within 120 seconds. Diagnostics:"
    docker compose ps
    docker compose logs --tail 50 api
    exit 4
}
Write-Host "    API is healthy."

docker compose ps
Write-Host @"

Stack ready:
  API   : http://localhost:8000
  Docs  : http://localhost:8000/docs
  Stop  : .\run.ps1 -Down   (or: docker compose down)

Auth and rate limiting are disabled by default; see .env to enable them.
"@ -ForegroundColor Green
exit 0
