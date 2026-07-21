@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Single entry point: installs anything missing, then starts the backend,
REM the frontend, and the browser.
REM   start.bat          - default port 5173
REM   start.bat 5199     - run the frontend on another port

REM Full paths - these tools are not always on a trimmed PATH.
REM Readiness checks use localhost, not 127.0.0.1: Vite binds IPv6 [::1]
REM only, so an IPv4 probe never answers even once it is up.
set NETSTAT=%SystemRoot%\System32\netstat.exe
set FINDSTR=%SystemRoot%\System32\findstr.exe
set CURL=%SystemRoot%\System32\curl.exe
set SLEEP=%SystemRoot%\System32\ping.exe

set BACKEND_PORT=8000
set FRONTEND_PORT=%1
if "%FRONTEND_PORT%"=="" set FRONTEND_PORT=5173

echo ============================================================
echo   Multi-Agent Medical Assessment
echo ============================================================
echo.

REM ---------------------------------------------------------------
REM 1. Prerequisites
REM ---------------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found.
    echo        Install Python 3.11 or newer from https://www.python.org/downloads/
    echo        During install, tick "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if !PYMAJOR! LSS 3 goto :oldpython
if !PYMAJOR! EQU 3 if !PYMINOR! LSS 11 goto :oldpython
goto :pythonok
:oldpython
echo ERROR: Python 3.11+ is required, but !PYVER! is installed.
echo        Get a newer build from https://www.python.org/downloads/
echo.
pause
exit /b 1
:pythonok

node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js was not found.
    echo        Install the LTS build from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 2. Install anything missing (skipped once installed)
REM ---------------------------------------------------------------
python -c "import fastapi, uvicorn, openai, pydantic_settings" >nul 2>&1
if errorlevel 1 (
    echo First run - installing Python packages, this takes a minute...
    python -m pip install --upgrade pip --quiet
    python -m pip install -e ".[dev]" --quiet
    if errorlevel 1 (
        echo.
        echo ERROR: Python package install failed. Run this to see why:
        echo            python -m pip install -e ".[dev]"
        echo.
        pause
        exit /b 1
    )
    echo       Done.
)

if not exist "frontend\node_modules" (
    echo First run - installing frontend packages, this takes a few minutes...
    pushd frontend
    call npm install --silent
    if errorlevel 1 (
        popd
        echo.
        echo ERROR: npm install failed. Run "npm install" in the frontend folder to see why.
        echo.
        pause
        exit /b 1
    )
    popd
    echo       Done.
)

REM ---------------------------------------------------------------
REM 3. Configuration
REM ---------------------------------------------------------------
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo.
    echo ============================================================
    echo   Almost there - your API key is needed.
    echo ============================================================
    echo.
    echo   A new .env file was just created in this folder and opened
    echo   for you. Replace this line:
    echo.
    echo       OPENAI_API_KEY=sk-your-key-here
    echo.
    echo   with your real OpenAI key, save it, then run start.bat again.
    echo.
    start "" notepad ".env"
    pause
    exit /b 1
)

%FINDSTR% /c:"OPENAI_API_KEY=sk-your-key-here" ".env" >nul 2>&1
if not errorlevel 1 (
    echo ERROR: The .env file still contains the placeholder API key.
    echo        Replace OPENAI_API_KEY=sk-your-key-here with your real key,
    echo        save the file, then run start.bat again.
    echo.
    start "" notepad ".env"
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 4. Ports
REM ---------------------------------------------------------------
REM Is anything already on the frontend port?
set FRONTEND_RUNNING=0
%NETSTAT% -ano | %FINDSTR% ":%FRONTEND_PORT% " | %FINDSTR% LISTENING >nul 2>&1
if not errorlevel 1 (
    REM Something is there. Is it us, or a different app?
    REM ConsensusPanel.jsx exists only in this project - another copy serves
    REM its index.html fallback instead, and the page title is identical in
    REM both, so the title cannot tell them apart.
    %CURL% -s -m 3 http://localhost:%FRONTEND_PORT%/src/components/ConsensusPanel.jsx 2>nul | %FINDSTR% /c:"Panel Agreement" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Port %FRONTEND_PORT% is in use by a different application.
        echo.
        echo        This is often another copy of this project. Close it first,
        echo        or start this one on its own port:
        echo.
        echo            start.bat 5199
        echo.
        pause
        exit /b 1
    )
    set FRONTEND_RUNNING=1
)

REM ---------------------------------------------------------------
REM 5. Backend
REM ---------------------------------------------------------------
%NETSTAT% -ano | %FINDSTR% ":%BACKEND_PORT% " | %FINDSTR% LISTENING >nul 2>&1
if not errorlevel 1 (
    echo [1/3] Backend already running on port %BACKEND_PORT% - reusing it.
) else (
    echo [1/3] Starting backend on port %BACKEND_PORT%...
    start "Multi-Agent Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn api.main:app --reload --port %BACKEND_PORT%"

    set /a TRIES=0
    :waitbackend
    %SLEEP% -n 3 127.0.0.1 >nul
    %CURL% -s -m 2 -o nul http://localhost:%BACKEND_PORT%/health >nul 2>&1
    if not errorlevel 1 goto backendup
    set /a TRIES+=1
    if !TRIES! GEQ 20 (
        echo.
        echo       ERROR: The backend did not start.
        echo       Check the "Multi-Agent Backend" window for the reason.
        echo.
        pause
        exit /b 1
    )
    goto waitbackend
    :backendup
    echo       Backend is up.
)

REM ---------------------------------------------------------------
REM 6. Frontend
REM ---------------------------------------------------------------
if "!FRONTEND_RUNNING!"=="1" (
    echo [2/3] Frontend already running on port %FRONTEND_PORT% - reusing it.
    goto frontendup
)

echo [2/3] Starting frontend on port %FRONTEND_PORT%...
REM --strictPort so it fails loudly instead of drifting to another port.
start "Multi-Agent Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev -- --port %FRONTEND_PORT% --strictPort"

set /a TRIES=0
:waitfrontend
%SLEEP% -n 3 127.0.0.1 >nul
%CURL% -s -m 2 -o nul http://localhost:%FRONTEND_PORT% >nul 2>&1
if not errorlevel 1 goto frontendup
set /a TRIES+=1
if !TRIES! GEQ 30 (
    echo.
    echo       ERROR: The frontend did not start.
    echo       Check the "Multi-Agent Frontend" window for the reason.
    echo.
    pause
    exit /b 1
)
goto waitfrontend
:frontendup
echo       Frontend is up.

REM ---------------------------------------------------------------
REM 7. Browser
REM ---------------------------------------------------------------
echo [3/3] Opening the browser...
start "" http://localhost:%FRONTEND_PORT%

echo.
echo ============================================================
echo   Running.
echo ============================================================
echo.
echo   App:      http://localhost:%FRONTEND_PORT%
echo   API docs: http://localhost:%BACKEND_PORT%/docs
echo.
echo   The backend and frontend each run in their own window.
echo   Closing those windows stops the app.
echo.
echo   This window can be closed.
echo.
pause
endlocal
