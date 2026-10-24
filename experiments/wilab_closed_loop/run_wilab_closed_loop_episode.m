function run_wilab_closed_loop_episode(configFile, traceFile, outputFolder, bridgeDir, episode, safeBudget, generationInterval, allocationPeriod, seed, controllerEpochs, safetyPayloadBytes, simulationTime, replayMode, replayFile, bwMHz, reservationInterval)
% Run one isolated WiLab episode with the Python action bridge enabled.
global WILAB_CLOSED_LOOP;
WILAB_CLOSED_LOOP.bridgeDir = bridgeDir;
WILAB_CLOSED_LOOP.episode = episode;
WILAB_CLOSED_LOOP.epoch = 0;
WILAB_CLOSED_LOOP.safeBudget = safeBudget;
WILAB_CLOSED_LOOP.lastControl = -Inf;
WILAB_CLOSED_LOOP.lastResultEpoch = -1;
WILAB_CLOSED_LOOP.instrument_enqueued_packets = 0;
WILAB_CLOSED_LOOP.instrument_enqueued_bytes = 0;
WILAB_CLOSED_LOOP.instrument_scheduled_attempts = 0;
WILAB_CLOSED_LOOP.instrument_pending_packets = 0;
WILAB_CLOSED_LOOP.radioEventLogging = strcmp(getenv('WILAB_RADIO_EVENT_LOGGING'),'1');
WILAB_CLOSED_LOOP.radio_events = struct([]);
warmupEnv = getenv('WILAB_WARMUP_SECONDS');
if ~isempty(warmupEnv)
    WILAB_CLOSED_LOOP.warmupSeconds = str2double(warmupEnv);
else
    WILAB_CLOSED_LOOP.warmupSeconds = 0.0;
end
WILAB_CLOSED_LOOP.controllerEpochs = 100;
if nargin < 7 || isempty(generationInterval), generationInterval = 0.1; end
if nargin < 8 || isempty(allocationPeriod), allocationPeriod = 0.1; end
WILAB_CLOSED_LOOP.allocationPeriod = allocationPeriod;
if nargin < 9 || isempty(seed), seed = 0; end
if nargin >= 10 && ~isempty(controllerEpochs), WILAB_CLOSED_LOOP.controllerEpochs = controllerEpochs; end
if nargin < 11 || isempty(safetyPayloadBytes), safetyPayloadBytes = 1000; end
if nargin < 12 || isempty(simulationTime), simulationTime = 20; end
if nargin < 13 || isempty(replayMode), replayMode = 'off'; end
if nargin < 14, replayFile = ''; end
% Corrected physical configuration: 20 MHz carrier and a 5 ms semi-persistent
% reservation, while allocationPeriod remains the 20 ms controller clock.
% Legacy callers omit these trailing arguments and keep 10 MHz / the
% allocation period.
if nargin < 15 || isempty(bwMHz), bwMHz = 10; end
if nargin < 16 || isempty(reservationInterval), reservationInterval = allocationPeriod; end

% This file lives in <repo>/experiments/wilab_closed_loop; three parents is repo.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
simroot = fullfile(root,'vendor','WiLabV2Xsim');
addpath(genpath(simroot));
% Put the isolated hook directory last at the front so its mainV2X and
% toolbox-free compatibility shims override the vendor entry points.
addpath(fileparts(mfilename('fullpath')),'-begin');
if strcmpi(replayMode,'record'), wilab_event_replay('setup_record',replayFile);
elseif strcmpi(replayMode,'replay'), wilab_event_replay('setup_replay',replayFile); end
WiLabV2Xsim(configFile,'outputFolder',outputFolder,'filenameTrace',traceFile, ...
    'Technology','5G-V2X','MCS_NR',11,'SCS_NR',15,'beaconSizeBytes',safetyPayloadBytes, ...
    'BwMHz',bwMHz,'reservationInterval',reservationInterval, ...
    'simulationTime',simulationTime,'allocationPeriod',allocationPeriod,'generationInterval',generationInterval, ...
    'seed',seed, ...
    'sizeSubchannel',10, ...
    'powerThresholdAutonomous',-126,'Raw',150,'FixedPdensity',false, ...
    'dcc_active',~strcmpi(getenv('WILAB_DCC_ACTIVE'),'0'),'cbrActive',true);
if strcmpi(replayMode,'record'), wilab_event_replay('flush'); end
end
