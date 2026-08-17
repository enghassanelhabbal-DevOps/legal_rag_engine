# Run the full local stack on Windows: backend in Docker, Streamlit UI locally
# Usage: Open PowerShell in repository root and run: .\run_local_windows.ps1
# This script is idempotent for typical dev use.

param(
    [string]$RepoRoot = "C:\Users\H Elhabbal\Downloads\legal_rag_engine.worktrees\devops-mlops-streamlit-integration",
    [int]$BackendPort = 8000,
    [int]$StreamlitPort = 8501
)

Set-StrictMode -Version Latest
Push-Location $RepoRoot

Write-Host "[1/7] Ensuring docker compose file present..."
if (-not (Test-Path -Path "$RepoRoot\docker-compose.yml")) {
    if (Test-Path -Path "$RepoRoot\docker-compose.example.yml") {
        Copy-Item -Path "$RepoRoot\docker-compose.example.yml" -Destination "$RepoRoot\docker-compose.yml" -Force
        Write-Host "Copied docker-compose.example.yml -> docker-compose.yml"
    } else {
        Write-Warning "docker-compose.example.yml not found. Please provide compose file."
    }
} else {
    Write-Host "docker-compose.yml already present"
}

Write-Host "[2/7] Writing .env (local test values)"
$envContent = @"
API_KEY=test-local-api-key
BACKEND_API_KEY=test-local-backend-key
BACKEND_URL=http://localhost:8000
ALLOW_LOCAL_MODEL_RUNTIME=0
HF_TOKEN=
OPENAI_API_KEY=
GEMINI_API_KEY=
DENSE_MODEL_NAME=BAAI/bge-m3
RERANKER_NAME=BAAI/bge-reranker-v2-m3
"@
$envContent | Set-Content -Path "$RepoRoot\.env" -Encoding UTF8
Write-Host ".env written (do not commit this file to git)."

Write-Host "[3/7] Building and starting backend (docker compose api)..."
# Build and start only the api service to reduce resources
$dockerCompose = "docker compose"
# Check docker available
try {
    & docker version > $null 2>&1
} catch {
    Write-Error "Docker not available in PATH. Install Docker Desktop and enable Linux containers. Aborting."
    Pop-Location
    exit 1
}

# Start api
$startCmd = "$dockerCompose up -d --build api"
Write-Host "Running: $startCmd"
Invoke-Expression $startCmd

Write-Host "Waiting for backend to respond on http://localhost:$BackendPort/v1/health (timeout 120s)"
$ok = $false
$start = Get-Date
while (((Get-Date) - $start).TotalSeconds -lt 120) {
    try {
        $resp = Invoke-RestMethod -Method Get -Uri "http://localhost:$BackendPort/v1/health" -Headers @{ 'x-api-key' = 'test-local-backend-key' } -TimeoutSec 5
        Write-Host "Backend health returned: $($resp | ConvertTo-Json -Depth 2)"
        $ok = $true
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}
if (-not $ok) {
    Write-Warning "Backend did not respond in time. Check docker logs: docker compose logs api --follow"
}

Write-Host "[4/7] Creating Python venv and installing requirements for Streamlit UI"
if (-not (Test-Path -Path "$RepoRoot\\.venv")) {
    python -m venv .venv
}
$activate = "$RepoRoot\\.venv\\Scripts\\Activate.ps1"
if (-not (Test-Path -Path $activate)) {
    Write-Warning "Virtualenv activate script not found at $activate. Ensure Python is installed and in PATH."
} else {
    Write-Host "Activating venv and installing requirements..."
    # Install requirements in a new process to avoid polluting this script's environment
    $pipInstall = "& `"$RepoRoot\\.venv\\Scripts\\python.exe`" -m pip install --upgrade pip && `"$RepoRoot\\.venv\\Scripts\\python.exe`" -m pip install -r requirements.txt"
    Write-Host $pipInstall
    Invoke-Expression $pipInstall
}

Write-Host "[5/7] Launching Streamlit UI in a new PowerShell window (localhost:$StreamlitPort)"
$streamlitCmd = "`"$RepoRoot\\.venv\\Scripts\\python.exe`" -m streamlit run `"$RepoRoot\\app.py`" --server.port $StreamlitPort"
$powershellArgs = "-NoExit -Command `$env:API_KEY='test-local-api-key'; `$env:BACKEND_URL='http://localhost:8000'; `$env:ALLOW_LOCAL_MODEL_RUNTIME='0'; $streamlitCmd"
Start-Process -FilePath "powershell.exe" -ArgumentList $powershellArgs

Write-Host "[6/7] Running a quick integration smoke POST to /v1/query"
try {
    $body = @{ query = 'ما هو تعريف البيانات الشخصية؟'; top_k = 3; jurisdiction='EG' } | ConvertTo-Json
    $resp = Invoke-RestMethod -Method Post -Uri "http://localhost:$BackendPort/v1/query" -Headers @{ 'x-api-key' = 'test-local-backend-key'; 'Content-Type' = 'application/json' } -Body $body -TimeoutSec 10
    Write-Host "Query response sample:"
    $resp | ConvertTo-Json -Depth 2 | Write-Host
} catch {
    Write-Warning "Integration query failed or timed out. You can inspect backend logs: docker compose logs api --tail 200"
}

Write-Host "[7/7] Done. Open http://localhost:$StreamlitPort in your browser."
Pop-Location
