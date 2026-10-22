$ErrorActionPreference = "Continue"
Set-Location "C:\Users\dash\.codex\worktrees\b2cd\5GProject3.0"
$env:PYTHONUNBUFFERED = "1"
$log = "sparc_performance_evaluation\results\raw\retry_density_d25.log"
$py = "C:\Users\dash\AppData\Local\Programs\Python\Python311\python.exe"
function W($m) { $line = "$(Get-Date -Format o) $m"; Add-Content -Path $log -Value $line; Write-Output $line }
W "retry waiter v4 start"
& $py -u sparc_performance_evaluation/scripts/run_paired_eval.py --output-root sparc_performance_evaluation/results/raw/density_d25 --seed-file sparc_performance_evaluation/protocol/final_test_seeds.json --density 25 --methods value_aware minfinish residual itora_inspired --checkpoint checkpoints/wilab_residual_ppo_96ep_va_relative.pt *>> $log 2>&1
$rc1 = $LASTEXITCODE; W "density_d25 rc=$rc1"
if ($rc1 -eq 0) { Set-Content "sparc_performance_evaluation\results\raw\density_d25\DONE" '{"recovered":true}' -Encoding utf8 }
& $py -u sparc_performance_evaluation/scripts/run_paired_eval.py --output-root sparc_performance_evaluation/results/raw/density_direct_d25 --seed-file sparc_performance_evaluation/protocol/final_test_seeds.json --density 25 --methods residual --checkpoint checkpoints/wilab_direct_ppo_96ep.pt --logit-mode direct *>> $log 2>&1
$rc2 = $LASTEXITCODE; W "density_direct_d25 rc=$rc2"
if ($rc2 -eq 0) { Set-Content "sparc_performance_evaluation\results\raw\density_direct_d25\DONE" '{"recovered":true}' -Encoding utf8 }
Remove-Item "sparc_performance_evaluation\results\aggregates\campaign_aggregate_DONE" -Force -ErrorAction SilentlyContinue
& $py -u sparc_performance_evaluation/scripts/aggregate_campaign.py *>> $log 2>&1
$rc3 = $LASTEXITCODE; W "re-aggregate rc=$rc3"
if ($rc3 -eq 0) { Set-Content "sparc_performance_evaluation\results\aggregates\campaign_aggregate_DONE" '{"recovered_after_d25":true}' -Encoding utf8 }
Set-Content "sparc_performance_evaluation\results\raw\density_d25_retry_DONE" (@{d25=$rc1; direct=$rc2; aggregate=$rc3} | ConvertTo-Json) -Encoding utf8
W "wrote dens25 retry gate; done"
