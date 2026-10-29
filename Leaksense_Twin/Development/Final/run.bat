@echo off
echo ============================================================
echo   LeakSense Twin - Cat C18 Engine Diagnostics
echo   Starting Backend + Frontend Servers
echo ============================================================
echo.

:: Kill existing backend if running
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5173" ^| find "LISTENING"') do taskkill /f /pid %%a 2>nul

:: Convert to short 8.3 paths to avoid & breaking cmd
for %%I in ("%~dp0backend") do set "BACKEND_SHORT=%%~sI"
for %%I in ("%~dp0frontend") do set "FRONTEND_SHORT=%%~sI"

:: Start Backend
echo [1/2] Starting Backend Server (FastAPI on port 8000)...
start "LeakSense Backend" cmd /k "cd /d %BACKEND_SHORT% && python main.py"
timeout /t 5 /nobreak > nul

:: Start Frontend
echo [2/2] Starting Frontend Dev Server (Vite on port 5173)...
start "LeakSense Frontend" cmd /k "cd /d %FRONTEND_SHORT% && node node_modules\vite\bin\vite.js build && node node_modules\vite\bin\vite.js preview --port 5173"
timeout /t 3 /nobreak > nul

echo.
echo ============================================================
echo   LeakSense Twin is running!
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8000/docs
echo ============================================================
echo.
echo Opening dashboard in browser...
start http://localhost:5173
