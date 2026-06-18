#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\pip.exe install -r requirements_lock.txt
}

$py = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = "$Root\src\python"

$torchOk = $true
try {
    & $py -c "import torch" 2>$null
    if ($LASTEXITCODE -ne 0) { $torchOk = $false }
} catch { $torchOk = $false }

if ($torchOk) {
    Write-Host "=== PyTorch pipeline ==="
    & $py tools\fetch_gm12878_hic.py --demo
    & $py tools\train.py --epochs 60 --ablation-rank
    & $py tools\verify_audit.py
} else {
    Write-Host "=== PyTorch unavailable — running NumPy smoke test ==="
    & $py tools\numpy_smoke.py
}

Write-Host "=== HGT-PSD PIPELINE COMPLETE ==="
