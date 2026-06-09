# Start all Legal Multi-Agent System services (Windows)
# Registry first, then leaf agents, then orchestrators

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Start-AgentService {
    param([string]$Name, [string]$Module)
    Write-Host "Starting $Name..."
    Start-Process -FilePath "uv" -ArgumentList "run", "python", "-m", $Module -WorkingDirectory $PSScriptRoot
}

Write-Host "Starting Registry service on port 10000..."
Start-AgentService "Registry" "registry"
Start-Sleep -Seconds 2

Start-AgentService "Tax Agent (10102)" "tax_agent"
Start-AgentService "Compliance Agent (10103)" "compliance_agent"
Start-Sleep -Seconds 3

Start-AgentService "Law Agent (10101)" "law_agent"
Start-Sleep -Seconds 3

Start-AgentService "Customer Agent (10100)" "customer_agent"

Write-Host ""
Write-Host "All services started in separate windows:"
Write-Host "  Registry:         http://localhost:10000"
Write-Host "  Customer Agent:   http://localhost:10100"
Write-Host "  Law Agent:        http://localhost:10101"
Write-Host "  Tax Agent:        http://localhost:10102"
Write-Host "  Compliance Agent: http://localhost:10103"
Write-Host ""
Write-Host "Wait ~10 seconds, then run:"
Write-Host "  uv run python test_client.py"
Write-Host "  .\start_demo.ps1   (web UI at http://localhost:10200)"
Write-Host ""
Write-Host "Close each service window (or Ctrl+C) to stop."
