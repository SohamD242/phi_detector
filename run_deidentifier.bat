@echo off
:: ========================================
:: CADIS PHI Detector - Automated Runner
:: ========================================

:: Change to the directory where this batch file is located
cd /d "%~dp0"

echo.
echo ========================================
echo CADIS PHI Detector
echo Protected Health Information Deidentification Tool
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.7+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python installation... OK
echo.

:: Check if virtual environment exists, if not create it
if not exist "venv\" (
    echo [2/4] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure you have Python 3.7+ with pip installed.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
) else (
    echo [2/4] Virtual environment found... OK
    echo.
)

:: Verify virtual environment is functional
if exist "venv\Scripts\python.exe" (
    :: Test if venv's Python actually works
    venv\Scripts\python.exe --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo WARNING: Virtual environment is corrupted or Python path is invalid.
        echo Deleting and recreating virtual environment...
        rmdir /s /q venv
        python -m venv venv
        if %errorlevel% neq 0 (
            echo ERROR: Failed to recreate virtual environment.
            pause
            exit /b 1
        )
        echo Virtual environment recreated successfully.
    )
)
echo.

:: Install dependencies using virtual environment's Python directly
echo [3/4] Installing/Updating dependencies...

:: Use the venv's Python executable directly (no activation needed)
venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
venv\Scripts\python.exe -m pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

:: Check if input folder has PDFs
if not exist "data\input_pdfs\*.pdf" (
    echo WARNING: No PDF files found in data\input_pdfs\ folder.
    echo Please add PDF files to the input_pdfs folder and run again.
    echo.
    pause
    exit /b 0
)

:: Run the deidentification script using venv's Python
echo [4/4] Running PHI Deidentification...
echo.
venv\Scripts\python.exe -m src.main

:: Check if execution was successful
if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo SUCCESS!
    echo Deidentified files are saved in:
    echo data\output_pdfs\
    echo ========================================
) else (
    echo.
    echo ========================================
    echo ERROR: Deidentification failed.
    echo Please check the error messages above.
    echo ========================================
)

echo.
echo Press any key to exit...
pause >nul
