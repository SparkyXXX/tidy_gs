@echo off
set "ORIGINAL_DIR=%cd%"

echo Switching to submodule directory...
cd .\submodules\diff-gaussian-rasterization\
if %errorlevel% neq 0 (
    echo Error: Failed to switch to submodule directory, please check the path
    exit /b 1
)

echo Cleaning old build artifacts...
if exist build (
    rmdir /s /q build
)
if exist dist (
    rmdir /s /q dist
)
for /d %%d in (*.egg-info) do (
    rmdir /s /q "%%d"
)
del /q diff_gaussian_rasterization\*.so
del /q diff_gaussian_rasterization\*.pyd

echo Starting CUDA extension compilation...
python setup.py build_ext --inplace
REM pip install -e .

if %errorlevel% equ 0 (
    echo Compilation completed successfully
) else (
    echo Error occurred during compilation
    exit /b 1
)

echo Returning to original directory...
cd "%ORIGINAL_DIR%"

echo Running render script...
python render.py -m ./data/Hub/output --skip_train

echo Operation completed
