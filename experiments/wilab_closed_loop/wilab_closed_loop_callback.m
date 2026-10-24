function action = wilab_closed_loop_callback(bridgeDir, episode, epoch, radioState, candidates, safeBudget)
% File-based MATLAB callback for the isolated WiLab/Python bridge.
% The caller must invoke this at the frozen controller boundary.
if ~exist(bridgeDir,'dir'), mkdir(bridgeDir); end
stateFile = fullfile(bridgeDir, sprintf('state_%06d_%06d.json', episode, epoch));
actionFile = fullfile(bridgeDir, sprintf('action_%06d_%06d.json', episode, epoch));
if exist(actionFile,'file'), delete(actionFile); end

state.episode = episode;
state.epoch = epoch;
state.time_s = radioState.time_s;
state.density = radioState.density;
state.cbr = radioState.cbr;
state.prr = radioState.prr;
if isfield(radioState,'prr_valid'), state.prr_valid = radioState.prr_valid; else, state.prr_valid = false; end
if isfield(radioState,'cumulative_prr'), state.cumulative_prr = radioState.cumulative_prr; else, state.cumulative_prr = radioState.prr; end
if isfield(radioState,'window_prr'), state.window_prr = radioState.window_prr; else, state.window_prr = radioState.prr; end
if isfield(radioState,'window_prr_long'), state.window_prr_long = radioState.window_prr_long; else, state.window_prr_long = state.window_prr; end
if isfield(radioState,'window_prr_long_valid'), state.window_prr_long_valid = radioState.window_prr_long_valid; else, state.window_prr_long_valid = false; end
if isfield(radioState,'window_attempts'), state.window_attempts = radioState.window_attempts; else, state.window_attempts = 0; end
if isfield(radioState,'window_duration_ms'), state.window_duration_ms = radioState.window_duration_ms; else, state.window_duration_ms = 0; end
state.p95_delay_ms = radioState.p95_delay_ms;
if isfield(radioState,'cbr_valid'), state.cbr_valid = radioState.cbr_valid; else, state.cbr_valid = false; end
if isfield(radioState,'delay_valid'), state.delay_valid = radioState.delay_valid; else, state.delay_valid = false; end
state.active_vehicles = radioState.active_vehicles;
state.radio_load = radioState.radio_load;
% Causal backlog observables (measured before this epoch's admission).
if isfield(radioState,'pending_packets'), state.pending_packets = radioState.pending_packets; else, state.pending_packets = 0; end
if isfield(radioState,'scheduled_attempts'), state.scheduled_attempts = radioState.scheduled_attempts; else, state.scheduled_attempts = 0; end
if isfield(radioState,'n_reassign'), state.n_reassign = radioState.n_reassign; else, state.n_reassign = 0; end
state.safe_budget = safeBudget;
state.candidates = candidates;
fid = fopen(stateFile,'w');
fwrite(fid,jsonencode(state),'char'); fclose(fid);

deadline = tic;
while ~exist(actionFile,'file')
    pause(0.001);
    if toc(deadline)>300, error('Timed out waiting for Python action'); end
end
while true
    fid = fopen(actionFile,'r');
    if fid >= 0
        raw = fread(fid,'*char')'; fclose(fid);
    else
        raw = '';
    end
    if ~isempty(strtrim(raw))
        try
            action = jsondecode(raw);
            break;
        catch
            % Atomic writers can briefly expose a zero-length/partial file.
        end
    end
    pause(0.001);
    if toc(deadline)>300, error('Timed out reading Python action'); end
end
end
