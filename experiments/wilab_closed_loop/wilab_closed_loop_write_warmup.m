function wilab_closed_loop_write_warmup(bridgeDir, episode, warmupSlot, radioState)
% Log safety-only warm-up measurements without creating an RL state.
if ~exist(bridgeDir,'dir'), mkdir(bridgeDir); end
row.warmup_slot = warmupSlot;
row.episode = episode;
row.time_s = radioState.time_s;
row.cbr = radioState.cbr;
row.cbr_valid = radioState.cbr_valid;
row.prr = radioState.prr;
row.prr_valid = radioState.prr_valid;
row.p95_ms = radioState.p95_delay_ms;
row.p95_valid = radioState.delay_valid;
fname = fullfile(bridgeDir, sprintf('warmup_%06d_%06d.json',episode,warmupSlot));
fid = fopen(fname,'w'); fwrite(fid,jsonencode(row),'char'); fclose(fid);
end
