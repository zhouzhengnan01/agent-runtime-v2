param(
  [string]$InputJsonPath = '',
  [string]$LogFile = '',
  [int]$Tail = 0
)

$ErrorActionPreference = 'Stop'

# ========== Tail mode ==========
if ($Tail -gt 0) {
  if ([string]::IsNullOrWhiteSpace($LogFile)) {
    throw 'Tail mode requires -LogFile'
  }
  if (!(Test-Path -LiteralPath $LogFile)) {
    Write-Host '[TAIL_WAIT] Log file not yet created'
    return
  }
  $lines = Get-Content -LiteralPath $LogFile -Tail $Tail -ErrorAction SilentlyContinue
  if ($lines) { $lines | Write-Host }
  $lastChunk = Get-Content -LiteralPath $LogFile -Tail 80 -ErrorAction SilentlyContinue
  if ($lastChunk -and ($lastChunk -join "`n" -match 'TRAINING_PROCESS_DONE')) {
    Write-Host '[TAIL_DONE]'
  }
  return
}

# ========== Run mode ==========
if ([string]::IsNullOrWhiteSpace($InputJsonPath)) {
  throw 'Run mode requires -InputJsonPath'
}
if (!(Test-Path -LiteralPath $InputJsonPath)) {
  throw "input.json not found: $InputJsonPath"
}

$config = Get-Content -Raw -Path $InputJsonPath | ConvertFrom-Json
$envName = $config.runtime.conda_env_name

if ([string]::IsNullOrWhiteSpace($envName)) {
  throw 'input.json missing runtime.conda_env_name'
}

$scriptPath = Join-Path $PSScriptRoot 'run_yolo_training.py'
if (!(Test-Path -LiteralPath $scriptPath)) {
  throw "Training script not found: $scriptPath"
}

$pyArgs = "-u `"$scriptPath`" --input `"$InputJsonPath`""

# ========== Background mode ==========
if (![string]::IsNullOrWhiteSpace($LogFile)) {
  $pyArgs += " --log-file `"$LogFile`""

  Write-Host '[BG] Launching training in background'
  Write-Host "[BG] conda env: $envName"
  Write-Host "[BG] log file: $LogFile"

  $logDir = Split-Path -Path $LogFile -Parent
  if ($logDir -and !(Test-Path -Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
  }

  $fullCmd = "conda run --no-capture-output -n $envName python $pyArgs"
  $tmpBat = [System.IO.Path]::GetTempFileName() + '.cmd'
  Set-Content -LiteralPath $tmpBat -Value "@echo off`r`n$fullCmd" -Encoding ASCII
  Write-Host "[BG] cmd: $fullCmd"

  $proc = Start-Process -FilePath 'cmd.exe' -ArgumentList "/c `"$tmpBat`"" -WindowStyle Hidden -PassThru

  $waited = 0
  while ($waited -lt 15 -and !(Test-Path -LiteralPath $LogFile)) {
    Start-Sleep -Seconds 1
    $waited++
  }

  if (Test-Path -LiteralPath $LogFile) {
    Write-Host "[BG] Training started (PID: $($proc.Id)), writing log"
  } else {
    Write-Host "[BG] Warning: log not created in 15s, check PID $($proc.Id)"
  }
  Write-Host '[BG] Use -LogFile <path> -Tail 30 to poll progress'
  return
}

# ========== Foreground mode (original blocking behavior) ==========
Write-Host "[INFO] conda env: $envName"
Write-Host "[INFO] config: $InputJsonPath"
Write-Host "[INFO] script: $scriptPath"

conda run --no-capture-output -n $envName python -u $scriptPath --input $InputJsonPath

if ($LASTEXITCODE -ne 0) {
  throw "Training failed, exit code: $LASTEXITCODE"
}

Write-Host '[INFO] Training completed'
