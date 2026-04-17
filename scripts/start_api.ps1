param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonPath = Join-Path $Root ".venv\Scripts\python.exe"
$ApiPath = Join-Path $Root "app\api.py"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python virtual environment not found at $PythonPath"
}

$oldProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like "python*" -and
    $_.CommandLine -and
    $_.CommandLine -like "*$Root*" -and
    ($_.CommandLine -like "*app\api.py*" -or $_.CommandLine -like "*app/api.py*")
}

if ($oldProcesses) {
    $ids = @($oldProcesses | Select-Object -ExpandProperty ProcessId)
    Write-Host "Stopping old API process(es): $($ids -join ', ')"
    Stop-Process -Id $ids -Force
    Start-Sleep -Seconds 1
}

$env:PORT = "$Port"

$EnvPath = Join-Path $Root ".env"
if (Test-Path -LiteralPath $EnvPath) {
    Get-Content -LiteralPath $EnvPath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }

        $name, $value = $line.Split("=", 2)
        $name = $name.Trim()
        $value = $value.Trim().Trim('"').Trim("'")
        if ($name) {
            Set-Item -Path "Env:$name" -Value $value
        }
    }
    Write-Host "Loaded environment variables from .env"
}

Write-Host "Starting O2C Pipeline API on http://127.0.0.1:$Port"
Write-Host "Press CTRL+C in this terminal to stop it."
& $PythonPath $ApiPath
