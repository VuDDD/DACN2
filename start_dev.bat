@echo off
title Retina Medical AI Platform - Dev Launcher
echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Retina AI Backend (8000)" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo Waiting 3 seconds before starting Frontend Vite server...
timeout /t 3 /nobreak >nul

echo Starting Vite Frontend on http://localhost:5173 ...
start "Retina AI Frontend (5173)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo Both servers launched. Open http://localhost:5173 in your browser.
