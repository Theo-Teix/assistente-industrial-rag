@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Ambiente nao encontrado. Rode primeiro 01_INSTALAR.bat
    pause
    exit /b 1
)

echo Verificando Ollama...
powershell -NoProfile -Command "try { Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo Iniciando o Ollama em segundo plano...
    start "" /min ollama serve
    timeout /t 3 /nobreak >nul
)

echo Abrindo o assistente no navegador...
".venv\Scripts\python.exe" -m streamlit run app\interface.py
