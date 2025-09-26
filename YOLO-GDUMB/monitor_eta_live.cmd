@echo off
title ETA Progress Monitor - YOLO-GDUMB (YOLOv12s)

echo.
echo ========================================
echo [92m[1mETA Progress Monitor - YOLO-GDUMB[0m
echo ========================================
echo.
echo [94mMonitoring real-time training progress...[0m
echo [94mETA estimates and iteration tracking[0m
echo.

:monitor_loop
if exist YOLO-GDUMB\optimal_output\eta_progress.log (
    echo [92m[1m003[0m ETA log found - starting live tail...
    powershell -Command "Get-Content 'YOLO-GDUMB\\optimal_output\\eta_progress.log' -Wait -Tail 20"
) else (
    echo [93mWaiting for training to start...[0m
    echo    Looking for: YOLO-GDUMB\optimal_output\eta_progress.log
    timeout /t 5 /nobreak >nul
    goto monitor_loop
)

echo.
echo ETA monitoring stopped.
pause


