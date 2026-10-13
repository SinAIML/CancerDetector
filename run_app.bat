@echo off
echo 🔬 Histopathologic Cancer Detection App
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install requirements
        pause
        exit /b 1
    )
) else (
    echo ✅ Streamlit already installed
)

echo.
echo 🚀 Starting the Cancer Detection App...
echo 📱 The app will open in your default web browser
echo 🛑 Press Ctrl+C to stop the app
echo.

REM Set Streamlit config to skip email prompt
set STREAMLIT_EMAIL_ENABLED=false

streamlit run streamlit_app.py --server.headless true

pause
