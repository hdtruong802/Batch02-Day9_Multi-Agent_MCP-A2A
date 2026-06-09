# Start demo UI (port 10200). Requires agents running via start_all.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$DemoPort = if ($env:DEMO_PORT) { [int]$env:DEMO_PORT } else { 10200 }
$DemoUrl = "http://localhost:$DemoPort"

function Test-DemoRunning {
    try {
        $resp = Invoke-WebRequest -Uri "$DemoUrl/api/health" -TimeoutSec 2 -UseBasicParsing
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Stop-PortListener {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) {
        $procId = $conn.OwningProcess
        Write-Host "Port $Port in use by PID $procId - stopping old process..."
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        return $true
    }
    return $false
}

if (Test-DemoRunning) {
    Write-Host "Demo UI is already running at $DemoUrl"
    Write-Host "Open that URL in your browser. To restart, close the other demo window first."
    exit 0
}

if (Stop-PortListener -Port $DemoPort) {
    Write-Host "Cleared stale process on port $DemoPort"
}

Write-Host "Starting Demo UI on $DemoUrl"
Write-Host "Ensure .env has: DEMO_TRACE_URL=$DemoUrl"
Write-Host 'And all agents are running: .\start_all.ps1'
Write-Host ""

uv run python -m demo
