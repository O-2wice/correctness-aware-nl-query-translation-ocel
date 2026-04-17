$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$processes = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like "python*" -and
    $_.CommandLine -and
    $_.CommandLine -like "*$Root*" -and
    ($_.CommandLine -like "*app\api.py*" -or $_.CommandLine -like "*app/api.py*")
}

if (-not $processes) {
    Write-Host "No O2C Pipeline API processes are running."
    exit 0
}

$ids = @($processes | Select-Object -ExpandProperty ProcessId)
Write-Host "Stopping O2C Pipeline API process(es): $($ids -join ', ')"
Stop-Process -Id $ids -Force
Write-Host "Stopped."
