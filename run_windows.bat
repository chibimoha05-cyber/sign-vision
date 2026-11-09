@echo off
setlocal enabledelayedexpansion
title Sign Language Vision Engine

echo ======================================================
echo   Sign Language Biomechanical Engine - Windows Setup
echo ======================================================

:: 1. Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/
    echo Make sure to check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

:: 2. Create virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [*] Installing dependencies from requirements.txt...
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
) else (
    call .venv\Scripts\activate.bat
)

:: 3. Launch Live Inspector
echo [*] Launching Real-Time Live Inspector...
python tools\live_inspector.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo Program exited with an error.
    pause
)

