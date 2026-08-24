@echo off
setlocal EnableExtensions

rem ============================================================
rem NebulonMind Installation Script (Windows)
rem Installs NebulonMind into a fixed location (%USERPROFILE%\.nebulonmind)
rem and creates a global 'nebulonmind' launcher in %USERPROFILE%\.local\bin.
rem ============================================================

set "REPO_URL=https://github.com/sathu08/NebulonMD.git"
set "BRANCH=main"

rem ------------------------------------------------------------
rem Check Git
rem ------------------------------------------------------------

where git >nul 2>nul
if errorlevel 1 (
    echo [NebulonMind][ERROR] Git is not installed. Please install Git first.
    exit /b 1
)

rem ------------------------------------------------------------
rem Clone or Update NebulonMind
rem ------------------------------------------------------------

echo [NebulonMind] NebulonMind repository:
echo [NebulonMind] %REPO_URL%
echo [NebulonMind] Target branch:
echo [NebulonMind] %BRANCH%

set "PROJECT_DIR=%USERPROFILE%\.nebulonmind"

echo [NebulonMind] Install directory:
echo [NebulonMind] %PROJECT_DIR%

if exist "%PROJECT_DIR%\.git" goto :update_repo

rem ---------- Fresh clone ----------
if exist "%PROJECT_DIR%" (
    echo [NebulonMind][ERROR] NebulonMind directory already exists but is not a Git repository: %PROJECT_DIR%
    exit /b 1
)

echo [NebulonMind] Cloning NebulonMind into:
echo [NebulonMind] %PROJECT_DIR%
echo [NebulonMind] Branch: %BRANCH%

mkdir "%PROJECT_DIR%"

git clone --branch %BRANCH% --single-branch %REPO_URL% "%PROJECT_DIR%"
if errorlevel 1 goto :error

echo [NebulonMind] Repository cloned successfully.
goto :movedir

:update_repo
echo [NebulonMind] NebulonMind repository already exists:
echo [NebulonMind] %PROJECT_DIR%

echo [NebulonMind] Fetching latest changes from branch '%BRANCH%'...
git fetch origin %BRANCH%
if errorlevel 1 goto :error

echo [NebulonMind] Switching to branch '%BRANCH%'...
git show-ref --verify --quiet refs/heads/%BRANCH%
if errorlevel 1 (
    git checkout -b %BRANCH% --track origin/%BRANCH%
) else (
    git checkout %BRANCH%
)
if errorlevel 1 goto :error

echo [NebulonMind] Updating branch '%BRANCH%'...
git pull --ff-only origin %BRANCH%
if errorlevel 1 goto :error

:movedir
rem ------------------------------------------------------------
rem Set Current Directory to NebulonMind
rem ------------------------------------------------------------

cd /d "%PROJECT_DIR%"
if errorlevel 1 goto :error

echo [NebulonMind] NebulonMind Home:
echo [NebulonMind] %PROJECT_DIR%

rem ------------------------------------------------------------
rem Verify Branch
rem ------------------------------------------------------------

for /f "delims=" %%i in ('git branch --show-current') do set "CURRENT_BRANCH=%%i"

if not "%CURRENT_BRANCH%"=="%BRANCH%" (
    echo [NebulonMind][ERROR] Expected branch '%BRANCH%', but currently on '%CURRENT_BRANCH%'.
    exit /b 1
)

echo [NebulonMind] Git branch:
echo [NebulonMind] %CURRENT_BRANCH%

rem ------------------------------------------------------------
rem Check / Install uv
rem ------------------------------------------------------------

where uv >nul 2>nul
if errorlevel 1 (
    echo [NebulonMind] uv is not installed.
    echo [NebulonMind] Installing uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto :error
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

where uv >nul 2>nul
if errorlevel 1 (
    echo [NebulonMind][ERROR] uv installation failed or uv is not available in PATH.
    exit /b 1
)

echo [NebulonMind] uv version:
uv --version

rem ------------------------------------------------------------
rem Check / Install Python 3.10
rem ------------------------------------------------------------

set "PYTHON_VERSION=3.10"

echo [NebulonMind] Checking Python %PYTHON_VERSION%...

uv python find %PYTHON_VERSION% >nul 2>nul
if errorlevel 1 (
    echo [NebulonMind] Python %PYTHON_VERSION% not found.
    echo [NebulonMind] Installing Python %PYTHON_VERSION%...
    uv python install %PYTHON_VERSION%
    if errorlevel 1 goto :error
)

for /f "delims=" %%i in ('uv python find %PYTHON_VERSION%') do set "PYTHON_PATH=%%i"

echo [NebulonMind] Using Python:
echo [NebulonMind] %PYTHON_PATH%

rem ------------------------------------------------------------
rem Install NebulonMind
rem
rem Requires a running NebulonDB backend (default localhost:6969)
rem at runtime; the install itself does not need one.
rem ------------------------------------------------------------

echo [NebulonMind] Installing NebulonMind dependencies...

uv sync --python %PYTHON_VERSION%
if errorlevel 1 goto :error

rem ------------------------------------------------------------
rem Install Global NebulonMind CLI
rem ------------------------------------------------------------

set "VENV_DIR=%PROJECT_DIR%\.venv"
set "CLI_PATH=%VENV_DIR%\Scripts\nebulonmind.exe"
set "BIN_DIR=%USERPROFILE%\.local\bin"
set "GLOBAL_CLI=%BIN_DIR%\nebulonmind.cmd"

if not exist "%CLI_PATH%" (
    echo [NebulonMind][ERROR] NebulonMind CLI was not created: %CLI_PATH%
    exit /b 1
)

echo [NebulonMind] Installing global NebulonMind CLI...

if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

> "%GLOBAL_CLI%" (
    echo @echo off
    echo set "NEBULONMD_HOME=%PROJECT_DIR%"
    echo "%CLI_PATH%" %%*
)

rem ------------------------------------------------------------
rem Configure ~/.local/bin in PATH
rem ------------------------------------------------------------

echo %PATH% | findstr /i /c:"%BIN_DIR%" >nul 2>nul
if errorlevel 1 (
    setx PATH "%BIN_DIR%;%PATH%" >nul
    if errorlevel 1 goto :error
)

set "PATH=%BIN_DIR%;%PATH%"

rem ------------------------------------------------------------
rem Verify NebulonMind CLI
rem ------------------------------------------------------------

where nebulonmind >nul 2>nul
if errorlevel 1 (
    echo [NebulonMind][ERROR] NebulonMind CLI was not installed correctly.
    exit /b 1
)

echo [NebulonMind] NebulonMind executable:
where nebulonmind

echo [NebulonMind] Testing NebulonMind CLI...
nebulonmind --help

rem ------------------------------------------------------------
rem Installation Complete
rem ------------------------------------------------------------

echo.
echo ============================================================
echo  NebulonMind Installation Complete
echo ============================================================
echo.
echo Repository      : %REPO_URL%
echo Branch          : %CURRENT_BRANCH%
echo Directory       : %PROJECT_DIR%
echo Python          : %PYTHON_PATH%
echo Virtual Env     : %VENV_DIR%
echo NebulonMind CLI : %BIN_DIR%\nebulonmind.cmd
echo NEBULONMD_HOME  : %PROJECT_DIR%
echo.
echo Next steps:
echo.
echo   1. Start NebulonDB first (backend):  nebulondb start
echo   2. Add NEBULONDB_USERNAME / NEBULONDB_PASSWORD to:
echo      %PROJECT_DIR%\.env
echo   3. Open a new terminal, then:
echo.
echo          nebulonmind start      API on http://localhost:9696
echo          nebulonmind            interactive chat TUI
echo.
echo ============================================================
exit /b 0

:error
echo [NebulonMind][ERROR] Installation failed.
exit /b 1
