function [sinrManagement,stationManagement,timeManagement,outputValues] = ...
            mainCV2XttiStarts(appParams,phyParams,timeManagement,sinrManagement,stationManagement,simParams,simValues,outParams,outputValues)
% a C-V2X TTI (time transmission interval) starts

% Compute the number of elapsed TTIs (Vittorio) (previously subframes)
% timeManagement.elapsedTime_subframes = floor((timeManagement.timeNow+1e-7)/phyParams.Tsf) + 1;
timeManagement.elapsedTime_TTIs = floor((timeManagement.timeNow+1e-7)/phyParams.TTI) + 1;

% %% Before v.5.4.16
% % BR adopted in the time domain (i.e., TTI)
% BRidT = ceil((stationManagement.BRid(:,1))/appParams.NbeaconsF);
% BRidT(stationManagement.BRid(:,1)<=0)=-1;
% 
% % Find IDs of vehicles that are currently transmitting
% stationManagement.transmittingIDsLTE = find(BRidT == (mod((timeManagement.elapsedTime_subframes-1),appParams.NbeaconsT)+1));
% % Remove those that do not have packets in the queue (occurs at the
% % beginning of the simulation)
% stationManagement.transmittingIDsLTE = stationManagement.transmittingIDsLTE.*(stationManagement.pckBuffer(stationManagement.transmittingIDsLTE)>0);
% stationManagement.transmittingIDsLTE(stationManagement.transmittingIDsLTE==0) = [];
% % The instant this packet was generated is saved - it is needed because a
% % new packet might be generated during this subframe and thus overwrite the
% % content of timeLastPacket - this was causing an inaccuracy in the KPIs
% stationManagement.transmittingFusedLTE = mod(stationManagement.BRid(stationManagement.transmittingIDsLTE,1)-1,appParams.NbeaconsF)+1;
% %%

%% From v.5.4.16
stationManagement.transmittingIDsCV2X = [];
stationManagement.hasTransmissionThisSlot=zeros(length(stationManagement.activeIDsCV2X),1);
iTransmitting = 1;
currentT = (mod((timeManagement.elapsedTime_TTIs-1),appParams.NbeaconsT)+1);
idLteHasPck = stationManagement.activeIDsCV2X(stationManagement.pckBuffer(stationManagement.activeIDsCV2X) >= 1);
% Apply the application-layer admission decision from the Python bridge.
% The packet remains a normal WiLab C-V2X packet; Mode-2 resource selection
% and PHY/MAC processing are still performed by WiLab.
global WILAB_CLOSED_LOOP;
if isempty(WILAB_CLOSED_LOOP), WILAB_CLOSED_LOOP = struct(); end
if ~isfield(WILAB_CLOSED_LOOP,'instrument_enqueued_packets'), WILAB_CLOSED_LOOP.instrument_enqueued_packets = 0; end
if ~isfield(WILAB_CLOSED_LOOP,'instrument_enqueued_bytes'), WILAB_CLOSED_LOOP.instrument_enqueued_bytes = 0; end
if ~isfield(WILAB_CLOSED_LOOP,'instrument_scheduled_attempts'), WILAB_CLOSED_LOOP.instrument_scheduled_attempts = 0; end
if ~isfield(WILAB_CLOSED_LOOP,'instrument_pending_packets'), WILAB_CLOSED_LOOP.instrument_pending_packets = 0; end
if ~isempty(WILAB_CLOSED_LOOP) && isfield(WILAB_CLOSED_LOOP,'action') && isfield(WILAB_CLOSED_LOOP.action,'allowed_sources')
    % Inject each admitted task's payload-faithful packet count once at the
    % start of its control interval.  The action remains visible to WiLab
    % for subsequent TTIs, but must not be re-added every TTI.
    injectNow = ~isfield(WILAB_CLOSED_LOOP,'lastInjectedEpoch') || ...
                ~isfield(WILAB_CLOSED_LOOP,'lastEpoch') || ...
                WILAB_CLOSED_LOOP.lastInjectedEpoch ~= WILAB_CLOSED_LOOP.lastEpoch;
    if injectNow && isfield(WILAB_CLOSED_LOOP.action,'source_packet_ids') && ...
            isfield(WILAB_CLOSED_LOOP.action,'source_packet_counts')
        sourceIds = double(WILAB_CLOSED_LOOP.action.source_packet_ids(:)) + 1;
        WILAB_CLOSED_LOOP.computation_source_ids = sourceIds(:)';
        packetCounts = double(WILAB_CLOSED_LOOP.action.source_packet_counts(:));
        n = min(numel(sourceIds), numel(packetCounts));
        for k = 1:n
            id = round(sourceIds(k));
            if id < 1 || id > numel(stationManagement.pckBuffer), continue; end
            wasEmpty = stationManagement.pckBuffer(id) < 1;
            stationManagement.pckBuffer(id) = stationManagement.pckBuffer(id) + max(0, round(packetCounts(k)));
            if wasEmpty, stationManagement.pckNextAttempt(id) = 1; end
        end
        WILAB_CLOSED_LOOP.instrument_enqueued_packets = WILAB_CLOSED_LOOP.instrument_enqueued_packets + sum(max(0, round(packetCounts(1:n))));
        if isfield(WILAB_CLOSED_LOOP.action,'source_payload_bytes')
            pb = double(WILAB_CLOSED_LOOP.action.source_payload_bytes(:));
            WILAB_CLOSED_LOOP.instrument_enqueued_bytes = WILAB_CLOSED_LOOP.instrument_enqueued_bytes + sum(max(0, pb(1:min(n,numel(pb)))));
        end
        WILAB_CLOSED_LOOP.lastInjectedEpoch = WILAB_CLOSED_LOOP.lastEpoch;
    else
        allowed = double(WILAB_CLOSED_LOOP.action.allowed_sources(:)) + 1; % legacy one-packet mode.
        allowed = allowed(allowed >= 1 & allowed <= numel(stationManagement.pckBuffer));
        wasEmpty = stationManagement.pckBuffer(allowed) < 1;
        stationManagement.pckBuffer(allowed) = max(stationManagement.pckBuffer(allowed), 1);
        stationManagement.pckNextAttempt(allowed(wasEmpty)) = 1;
    end
    idLteHasPck = stationManagement.activeIDsCV2X(stationManagement.pckBuffer(stationManagement.activeIDsCV2X) >= 1);
end
for idLte = idLteHasPck'    
    % A source may now have several queued computation packets, while the
    % configured Mode-2 reservation can have only one BR column.  Reuse the
    % valid reservation column for the next queued packet instead of indexing
    % past BRid's second dimension.
    attemptToDo = mod(stationManagement.pckNextAttempt(idLte)-1, size(stationManagement.BRid,2)) + 1;
    if ceil((stationManagement.BRid(idLte,attemptToDo))/appParams.NbeaconsF)==currentT
        stationManagement.transmittingIDsCV2X(iTransmitting) = idLte;
        stationManagement.transmittingFusedLTE(iTransmitting) = mod((stationManagement.BRid(idLte,attemptToDo))-1,appParams.NbeaconsF)+1;
        iTransmitting = iTransmitting + 1;
        % DEBUG TX-RX
        % printDebugTxRx(timeManagement.timeNow,idLte,'NR Tx started',stationManagement,sinrManagement,outParams);
    end
end
% hasTransmissionThisSlot introduced from version 6.2
stationManagement.hasTransmissionThisSlot(stationManagement.transmittingIDsCV2X)=1;
WILAB_CLOSED_LOOP.instrument_scheduled_attempts = WILAB_CLOSED_LOOP.instrument_scheduled_attempts + numel(stationManagement.transmittingIDsCV2X);
WILAB_CLOSED_LOOP.instrument_pending_packets = sum(stationManagement.pckBuffer(:));
%%

timeManagement.timeGeneratedPacketInTxLTE(stationManagement.transmittingIDsCV2X) = timeManagement.timeLastPacket(stationManagement.transmittingIDsCV2X);

if ~isempty(stationManagement.transmittingIDsCV2X)     
    % Find index of vehicles that are currently transmitting
    stationManagement.indexInActiveIDsOnlyLTE_OfTxLTE = zeros(length(stationManagement.transmittingIDsCV2X),1);
    stationManagement.indexInActiveIDs_OfTxLTE = zeros(length(stationManagement.transmittingIDsCV2X),1);
    for ix = 1:length(stationManagement.transmittingIDsCV2X)
        stationManagement.indexInActiveIDsOnlyLTE_OfTxLTE(ix) = find(stationManagement.activeIDsCV2X == stationManagement.transmittingIDsCV2X(ix));
        stationManagement.indexInActiveIDs_OfTxLTE(ix) = find(stationManagement.activeIDs == stationManagement.transmittingIDsCV2X(ix));
    end
end

% Initialization of the received power
[sinrManagement] = initLastPowerCV2X(timeManagement,stationManagement,sinrManagement,simParams,appParams,phyParams);

% COEXISTENCE IN THE SAME BAND
if simParams.technology == constants.TECH_COEX_STD_INTERF      
    [timeManagement,stationManagement,sinrManagement,outputValues] = coexistenceAtLTEsubframeStart(timeManagement,sinrManagement,stationManagement,appParams,simParams,simValues,phyParams,outParams,outputValues);    
end
    
% Remove the packet from the queue
% the packet is removed from the queue after the last transmission.
% If the last transmission is disabled (BRids=-1) the packet remains in the queue.
% At the next generation an of overflow will occur, which will bring back
% the pckBuffer to one and properly account for the correct/incorrect reception of the packet.
% This has been done to allow the possibility of allowing the possibility
% of triggering retransmissions
if ~isempty(stationManagement.transmittingIDsCV2X)
    stationManagement.pckTxOccurring(stationManagement.transmittingIDsCV2X) = stationManagement.pckNextAttempt(stationManagement.transmittingIDsCV2X);
    stationManagement.pckNextAttempt(stationManagement.transmittingIDsCV2X) = stationManagement.pckNextAttempt(stationManagement.transmittingIDsCV2X) + 1;
    txIDlastTx = stationManagement.transmittingIDsCV2X(stationManagement.pckNextAttempt(stationManagement.transmittingIDsCV2X)>stationManagement.cv2xNumberOfReplicas(stationManagement.transmittingIDsCV2X));
    stationManagement.pckBuffer(txIDlastTx) = stationManagement.pckBuffer(txIDlastTx)-1;
    % Start the next queued packet at its first replica/reservation attempt.
    stationManagement.pckNextAttempt(txIDlastTx) = 1;
    % reset of pckReceive and cumulativeSINR
    stationManagement.pckReceived(:,stationManagement.transmittingIDsCV2X(stationManagement.pckTxOccurring(stationManagement.transmittingIDsCV2X)==1)) = 0;
    sinrManagement.cumulativeSINR(:,stationManagement.transmittingIDsCV2X(stationManagement.pckTxOccurring(stationManagement.transmittingIDsCV2X)==1)) = 0;
end

