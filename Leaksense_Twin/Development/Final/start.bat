@echo off
echo ============================================================
echo   LeakSense Twin - Cat C18 Engine Diagnostics v3
echo   Starting Backend + Frontend Servers
echo ============================================================
echo.

:: Kill any existing servers on those ports
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING" 2^>nul') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5173" ^| find "LISTENING" 2^>nul') do taskkill /f /pid %%a 2>nul
timeout /t 1 /nobreak > nul

:: Convert to short 8.3 paths to avoid & breaking cmd
for %%I in ("%~dp0backend") do set "BACKEND_SHORT=%%~sI"
for %%I in ("%~dp0frontend") do set "FRONTEND_SHORT=%%~sI"

:: Start Backend (FastAPI / uvicorn)
echo [1/2] Starting Backend Server (FastAPI on port 8000)...
start "LeakSense Backend" cmd /k "cd /d %BACKEND_SHORT% && python main.py"
timeout /t 5 /nobreak > nul

:: Start Frontend (Vite dev server)
echo [2/2] Starting Frontend Dev Server (Vite on port 5173)...
start "LeakSense Frontend" cmd /k "cd /d %FRONTEND_SHORT% && npm run dev"
timeout /t 4 /nobreak > nul

echo.
echo ============================================================
echo   LeakSense Twin is RUNNING!
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8000/docs
echo ============================================================
echo.
echo Opening dashboard in browser...
start http://localhost:5173
