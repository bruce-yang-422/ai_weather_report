$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
} else {
    Write-Error "找不到可用的 Python。請先安裝 Python，或建立 .venv。"
}

try {
    if ($pythonExe -eq "py") {
        & $pythonExe -3 "scripts/weather.py"
    } else {
        & $pythonExe "scripts/weather.py"
    }
} catch {
    Write-Error "執行 weather.py 失敗: $($_.Exception.Message)"
}
