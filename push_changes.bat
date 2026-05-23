@echo off
setlocal

set "COMMIT_MESSAGE=%~1"
if "%COMMIT_MESSAGE%"=="" set "COMMIT_MESSAGE=Update project code"

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo This directory is not a git repository.
    echo Run this file from the project root after git init / clone.
    exit /b 1
)

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

git diff --cached --quiet
if not errorlevel 1 (
    echo Nothing to commit.
    exit /b 0
)

echo Creating commit: %COMMIT_MESSAGE%
git commit -m "%COMMIT_MESSAGE%"
if errorlevel 1 (
    echo git commit failed.
    exit /b 1
)

echo Pushing current branch...
git push
if errorlevel 1 (
    echo git push failed.
    exit /b 1
)

echo Done.
