#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Installs WSL2 with Ubuntu 22.04 for LLM serving.
.DESCRIPTION
    This script enables WSL2 features and installs Ubuntu 22.04.
    NVIDIA GPU passthrough is automatic with Windows driver 581.95 / CUDA 13.0.
    No GPU driver installation is needed inside WSL.
.NOTES
    Run this script as Administrator in PowerShell.
#>

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  WSL2 Installation for LLM Serving Setup"   -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- Check if WSL2 is already installed and working ---
$wslInstalled = $false
try {
    $wslStatus = wsl --status 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] WSL2 is already installed." -ForegroundColor Green
        $wslInstalled = $true
    }
} catch {
    $wslInstalled = $false
}

# --- Check if Ubuntu-22.04 is already installed ---
$ubuntuInstalled = $false
if ($wslInstalled) {
    $distros = wsl --list --quiet 2>&1
    if ($distros -match "Ubuntu-22.04") {
        Write-Host "[OK] Ubuntu-22.04 is already installed." -ForegroundColor Green
        $ubuntuInstalled = $true
    }
}

if ($wslInstalled -and $ubuntuInstalled) {
    Write-Host ""
    Write-Host "WSL2 and Ubuntu-22.04 are already set up." -ForegroundColor Green
    Write-Host "You can proceed to run setup_environment.sh inside WSL2." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  wsl -d Ubuntu-22.04" -ForegroundColor White
    Write-Host "  cd /mnt/c/... (navigate to project)" -ForegroundColor White
    Write-Host "  bash setup/setup_environment.sh" -ForegroundColor White
    exit 0
}

# --- Enable Windows features ---
Write-Host ""
Write-Host "Enabling Windows Subsystem for Linux..." -ForegroundColor Yellow
$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
if ($wslFeature.State -ne "Enabled") {
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart | Out-Null
    Write-Host "[OK] Windows Subsystem for Linux enabled." -ForegroundColor Green
} else {
    Write-Host "[OK] Windows Subsystem for Linux already enabled." -ForegroundColor Green
}

Write-Host "Enabling Virtual Machine Platform..." -ForegroundColor Yellow
$vmFeature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
if ($vmFeature.State -ne "Enabled") {
    Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart | Out-Null
    Write-Host "[OK] Virtual Machine Platform enabled." -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual Machine Platform already enabled." -ForegroundColor Green
}

# --- Set WSL2 as default version ---
Write-Host "Setting WSL2 as default version..." -ForegroundColor Yellow
wsl --set-default-version 2 2>&1 | Out-Null
Write-Host "[OK] WSL2 set as default." -ForegroundColor Green

# --- Install Ubuntu-22.04 ---
if (-not $ubuntuInstalled) {
    Write-Host ""
    Write-Host "Installing Ubuntu-22.04 (this may take a few minutes)..." -ForegroundColor Yellow
    wsl --install -d Ubuntu-22.04 --no-launch
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Ubuntu-22.04 installation initiated." -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Install command returned non-zero. A reboot may be required first." -ForegroundColor Yellow
    }
}

# --- Print next steps ---
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  NEXT STEPS"                                 -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. REBOOT your computer to complete WSL2 installation." -ForegroundColor Yellow
Write-Host ""
Write-Host "2. After reboot, open Ubuntu-22.04 from the Start Menu" -ForegroundColor Yellow
Write-Host "   (or run 'wsl -d Ubuntu-22.04' in a terminal)." -ForegroundColor Yellow
Write-Host "   You will be prompted to create a UNIX username and password." -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Inside WSL2, navigate to the project and run the environment setup:" -ForegroundColor Yellow
Write-Host ""
Write-Host '   cd /mnt/c/Users/soujanyar/"OneDrive - NVIDIA Corporation"/Documents/fun_projects/llm_user' -ForegroundColor White
Write-Host "   bash setup/setup_environment.sh" -ForegroundColor White
Write-Host ""
Write-Host "NOTE: NVIDIA GPU driver on Windows (581.95 / CUDA 13.0) already" -ForegroundColor Green
Write-Host "supports GPU passthrough to WSL2. No driver install is needed inside WSL." -ForegroundColor Green
Write-Host ""
