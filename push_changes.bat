@echo off
setlocal

set "REPO_URL=https://github.com/Saddiedd/ML.git"
set "REMOTE_NAME=origin"
set "COMMIT_MESSAGE=%~1"
if "%COMMIT_MESSAGE%"=="" set "COMMIT_MESSAGE=Update project code"

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo This directory is not a git repository.
    echo Initialize it first:
    echo   git init
    exit /b 1
)

for /f "delims=" %%b in ('git branch --show-current') do set "BRANCH_NAME=%%b"
if "%BRANCH_NAME%"=="" (
    set "BRANCH_NAME=main"
    echo No current branch detected. Using: %BRANCH_NAME%
)

git remote get-url %REMOTE_NAME% >nul 2>&1
if errorlevel 1 (
    echo Adding remote %REMOTE_NAME%: %REPO_URL%
    git remote add %REMOTE_NAME% %REPO_URL%
) else (
    echo Updating remote %REMOTE_NAME%: %REPO_URL%
    git remote set-url %REMOTE_NAME% %REPO_URL%
)

if errorlevel 1 (
    echo Failed to configure remote.
    exit /b 1
)

echo Remote configuration:
git remote -v

echo.
echo Staging project files without datasets, labs, generated models, and virtual environment...

git add -- . ^
    ":(exclude)train/**" ^
    ":(exclude)test/**" ^
    ":(exclude)data/**" ^
    ":(exclude)preprocessed/**" ^
    ":(exclude)results/**" ^
    ":(exclude)models/**" ^
    ":(exclude)venv/**" ^
    ":(exclude)__pycache__/**" ^
    ":(exclude)lab1_normalization/**" ^
    ":(exclude)lab2_processing/**" ^
    ":(exclude)lab3_analysis/**" ^
    ":(exclude)lab4_PCA/**" ^
    ":(exclude)lab5_frequency_analysis/**" ^
    ":(exclude)lab6_fpm_fpg/**" ^
    ":(exclude)lab7_classification/**" ^
    ":(exclude)lab8_clasterisation/**" ^
    ":(exclude)*.7z" ^
    ":(exclude)*.zip" ^
    ":(exclude)*.rar" ^
    ":(exclude)trainLabels.csv" ^
    ":(exclude)sampleSubmission.csv"

if errorlevel 1 (
    echo git add failed.
    exit /b 1
)

echo.
echo Staged files:
git diff --cached --name-status

git diff --cached --quiet
if not errorlevel 1 (
    echo Nothing to commit.
    exit /b 0
)

echo.
echo Creating commit: %COMMIT_MESSAGE%
git commit -m "%COMMIT_MESSAGE%"
if errorlevel 1 (
    echo git commit failed.
    exit /b 1
)

echo.
echo Pushing branch %BRANCH_NAME% to %REMOTE_NAME%...
git push -u %REMOTE_NAME% %BRANCH_NAME%
if errorlevel 1 (
    echo git push failed.
    exit /b 1
)

echo Done.
