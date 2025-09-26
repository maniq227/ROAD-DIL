Windows one-liners to monitor the new YOLO-GDUMB runs.

Tail ETA log with epoch overlay:

```powershell
powershell -File Misc\monitor_eta_with_epochs.ps1 `
  -Path "YOLO-GDUMB\optimal_output\eta_progress.log" `
  -Total 20 `
  -Out "YOLO-GDUMB\optimal_output\eta_progress_enhanced.log"
```

Plain tail:

```powershell
Get-Content YOLO-GDUMB\optimal_output\eta_progress.log -Wait -Tail 50
```

TensorBoard:

```powershell
tensorboard --logdir YOLO-GDUMB\optimal_output\tensorboard_logs --port 6012
```


