[CmdletBinding()]
param(
    [switch]$StartServer,
    [switch]$CpuOnly,
    [switch]$WithOcr,
    [switch]$WithTts,
    [switch]$WithTraining,
    [ValidateSet("1", "2")]
    [string]$RequiredReviewers = "1",
    [string]$TesseractPath,
    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if ($StartServer) {
    $portListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($portListener) {
        throw "Port $Port is already in use by process $($portListener.OwningProcess). Stop it or retry with -Port <free-port>."
    }
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "Python 3.10 or newer was not found on PATH. Install Python, reopen PowerShell, and retry."
}

$pythonLauncher = $pythonCommand.Source
$pythonArguments = @()
$environmentName = ".venv"
$pythonVersion = & $pythonLauncher --version 2>&1
if ($pythonVersion -notmatch "Python (\d+)\.(\d+)") {
    throw "Could not determine the Python version: $pythonVersion"
}
$pythonMajor = [int]$Matches[1]
$pythonMinor = [int]$Matches[2]
if ($pythonMajor -lt 3 -or ($pythonMajor -eq 3 -and $pythonMinor -lt 10)) {
    throw "Python 3.10 or newer is required. Detected: $pythonVersion"
}
if ($pythonMajor -eq 3 -and $pythonMinor -ge 12) {
    Write-Warning "This project recommends Python 3.10 or 3.11. Continuing with $pythonVersion; check package compatibility if installation fails."
}

if ($WithTts) {
    if ($pythonMajor -gt 3 -or ($pythonMajor -eq 3 -and $pythonMinor -gt 11)) {
        $pythonLauncherCommand = Get-Command py -ErrorAction SilentlyContinue
        if (-not $pythonLauncherCommand) {
            throw "TTS dependencies require Python 3.11 on this machine. Install Python 3.11 and retry with the Windows py launcher."
        }
        $pythonLauncher = $pythonLauncherCommand.Source
        $pythonArguments = @("-3.11")
        $pythonVersion = & $pythonLauncher @pythonArguments --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "TTS dependencies require Python 3.11 here. Install Python 3.11 and retry."
        }
    }
    $environmentName = ".venv-tts"
    Write-Host "Using the dedicated .venv-tts environment for TTS compatibility ($pythonVersion)."
}

$venvDirectory = Join-Path $projectRoot $environmentName
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating project virtual environment..."
    & $pythonLauncher @pythonArguments -m venv $venvDirectory
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

function Invoke-VenvPython {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: .venv Python $($Arguments -join ' ')"
    }
}

Write-Host "Upgrading pip..."
Invoke-VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")

if ($CpuOnly) {
    Write-Host "Installing CPU-only PyTorch..."
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
}

Write-Host "Installing core requirements..."
Invoke-VenvPython -Arguments @("-m", "pip", "install", "-r", "requirements.txt")

if ($WithOcr) {
    Write-Host "Installing optional OCR Python packages..."
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "-r", "requirements-ocr.txt")
    if ($TesseractPath) {
        if (-not (Test-Path $TesseractPath -PathType Leaf)) {
            throw "Tesseract executable was not found at: $TesseractPath"
        }
        $env:BS_TESSERACT_CMD = (Resolve-Path $TesseractPath).Path
    } elseif (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
        Write-Warning "Install Tesseract OCR separately and add it to PATH, or pass -TesseractPath. Install the hin and/or eng language data as needed."
    }
}

if ($WithTts) {
    Write-Host "Installing optional TTS packages..."
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "-r", "requirements-tts.txt")
}

if ($WithTraining) {
    Write-Host "Installing optional training packages..."
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "-r", "requirements-training.txt")
}

if ($StartServer) {
    function Read-ReviewerPin {
        param([Parameter(Mandatory)][string]$Prompt)
        while ($true) {
            $securePin = Read-Host -Prompt $Prompt -AsSecureString
            $plainPin = [System.Net.NetworkCredential]::new("", $securePin).Password
            if ($plainPin.Length -ge 8) { return $plainPin }
            Write-Warning "Passcodes must contain at least 8 characters. Try again."
        }
    }

    if ($WithTts -and [string]::IsNullOrWhiteSpace($env:HF_TOKEN)) {
        $secureToken = Read-Host -Prompt "Hugging Face read token for TTS model access" -AsSecureString
        $plainToken = [System.Net.NetworkCredential]::new("", $secureToken).Password
        if ([string]::IsNullOrWhiteSpace($plainToken)) {
            throw "A Hugging Face read token is required when starting with -WithTts."
        }
        $env:HF_TOKEN = $plainToken
        Remove-Variable secureToken, plainToken
    }

    $teacherPin = Read-ReviewerPin -Prompt "Teacher passcode (8+ characters)"
    do {
        $expertPin = Read-ReviewerPin -Prompt "Community expert passcode (different, 8+ characters)"
        if ($teacherPin -ceq $expertPin) {
            Write-Warning "The passcodes must be different."
        }
    } while ($teacherPin -ceq $expertPin)

    $env:BS_TEACHER_PIN = $teacherPin
    $env:BS_EXPERT_PIN = $expertPin
    $env:BS_REQUIRED_REVIEWERS = $RequiredReviewers
    $env:BS_MOCK = "0"
    $env:BS_PORT = [string]$Port

    Write-Host "Starting Bhasha Setu in real translation mode at http://127.0.0.1:$Port"
    Write-Host "The first startup may download the NLLB model. Keep this window open."
    & $venvPython (Join-Path $projectRoot "api.py")
    if ($LASTEXITCODE -ne 0) { throw "Bhasha Setu exited with code $LASTEXITCODE." }
} else {
    Write-Host "Setup complete. Start later with: $venvPython .\api.py"
    Write-Host "Set BS_TEACHER_PIN, BS_EXPERT_PIN, and BS_REQUIRED_REVIEWERS in that PowerShell session first."
    if ($WithOcr -and $TesseractPath) {
        Write-Host "Tesseract path for a future session: `$env:BS_TESSERACT_CMD = `"$env:BS_TESSERACT_CMD`""
    }
    if ($env:HF_TOKEN) {
        Write-Host "HF_TOKEN is available in this session and will be inherited by the server."
    }
}
