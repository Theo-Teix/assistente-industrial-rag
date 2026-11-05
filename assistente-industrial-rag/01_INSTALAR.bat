@echo off
setlocal
cd /d "%~dp0"
echo ==========================================
echo   INSTALADOR - Assistente Industrial RAG
echo ==========================================

rem Usa o Python 3.12 se existir; senao, o "python" do PATH
set PY=python
py -3.12 --version >nul 2>&1 && set PY=py -3.12
echo Usando: %PY%

if not exist ".venv\Scripts\python.exe" (
    echo Criando ambiente virtual .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo ERRO: nao consegui criar o .venv. Instale o Python 3.12 e marque "Add Python to PATH".
        pause
        exit /b 1
    )
)

echo Instalando bibliotecas (pode demorar alguns minutos)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERRO na instalacao. Verifique a internet e rode de novo.
    pause
    exit /b 1
)

echo.
echo INSTALACAO CONCLUIDA. Agora rode 02_INICIAR.bat
pause
