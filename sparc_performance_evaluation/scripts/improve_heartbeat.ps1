while ($true) {
  Start-Sleep -Seconds 300
  Write-Output 'AGENT_LOOP_TICK_wilab_campaign {"prompt":"Check WiLab improve queue: current train/FT progress, FAIL/hang, staged 96/+48/LAST48 gate; brief report; intervene only if hung/failed."}'
}
