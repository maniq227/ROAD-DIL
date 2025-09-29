@echo off
title TensorBoard Monitor - YOLO-GDUMB

echo.
echo ========================================
echo TensorBoard Monitor - YOLO-GDUMB
echo ========================================
echo.
echo Starting TensorBoard on http://localhost:6012
echo Logdir: YOLO-GDUMB\optimal_output\tensorboard_logs
echo.

tensorboard --logdir YOLO-GDUMB\optimal_output\tensorboard_logs --port 6012

echo.
echo TensorBoard stopped.
pause


