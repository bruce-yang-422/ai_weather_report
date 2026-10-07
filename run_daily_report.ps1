$ErrorActionPreference = 'Stop'
$reportRoot = $PSScriptRoot
$pythonExe = Join-Path $reportRoot '.venv\Scripts\python.exe'
$logDirectory = Join-Path $reportRoot 'output\scheduled'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory ((Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Set-Location -LiteralPath $reportRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
try {
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        throw "Python virtual environment missing: $pythonExe"
    }
    & $pythonExe (Join-Path $reportRoot 'main.py') > $logPath 2>&1
    $reportExitCode = $LASTEXITCODE
    Add-Content -LiteralPath $logPath -Value "`nExit code: $reportExitCode" -Encoding UTF8
    exit $reportExitCode
} catch {
    Add-Content -LiteralPath $logPath -Value $_.ToString() -Encoding UTF8
    exit 1
}
