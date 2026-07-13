@echo off
setlocal
where py.exe >nul 2>nul
if not errorlevel 1 (
    py.exe -3 "%~dp0show_calibration_target.py" %*
) else (
    python.exe "%~dp0show_calibration_target.py" %*
)
if errorlevel 1 (
    echo Failed to display the calibration target.
    echo Try: python "%~dp0show_calibration_target.py" --list-monitors
    pause
)
