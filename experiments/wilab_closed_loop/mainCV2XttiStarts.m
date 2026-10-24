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
% Read the fate of the packets whose last replica was sent in the previous
% TTI, before WiLab resets pckReceived for packets starting now.  Then bring
% the parallel tag FIFO back in sync with pckBuffer (queue overflow or
% vehicle exit drops packets without completing them).
if ~isempty(WILAB_CLOSED_LOOP)
    captureEpoch = 0;
    if isfield(WILAB_CLOSED_LOOP,'epoch'), captureEpoch = WILAB_CLOSED_LOOP.epoch; end
    % Read the fate of packets whose last replica was sent in the previous
    % TTI, then bring the tag FIFO back in sync with WiLab's queue.
    wilab_closed_loop_packet_accounting('capture', stationManagement, ...
        timeManagement.timeNow, captureEpoch);
    wilab_closed_loop_packet_accounting('reconcile', stationManagement, ...
        timeManagement.timeNow);
end
if ~isempty(WILAB_CLOSED_LOOP) && isfield(WILAB_CLOSED_LOOP,'action')
    % Register each admitted task's payload once per control interval.  The
    % payload then waits in the accounting layer's pending queue; it is never
    % dumped into WiLab's single-packet pckBuffer as a burst.
    injectNow = ~isfield(WILAB_CLOSED_LOOP,'lastInjectedEpoch') || ...
                ~isfield(WILAB_CLOSED_LOOP,'lastEpoch') || ...
                WILAB_CLOSED_LOOP.lastInjectedEpoch ~= WILAB_CLOSED_LOOP.lastEpoch;
    if injectNow
        stationManagement = wilab_closed_loop_packet_accounting('register', ...
            WILAB_CLOSED_LOOP.action, stationManagement, timeManagement.timeNow);
        WILAB_CLOSED_LOOP.lastInjectedEpoch = WILAB_CLOSED_LOOP.lastEpoch;
        if isfield(WILAB_CLOSED_LOOP.action,'source_wilab_ids')
            sourceIds = double(WILAB_CLOSED_LOOP.action.source_wilab_ids(:))';
        elseif isfield(WILAB_CLOSED_LOOP.action,'source_packet_ids')
            sourceIds = double(WILAB_CLOSED_LOOP.action.source_packet_ids(:))' + 1;
        else
            sourceIds = [];
        end
        WILAB_CLOSED_LOOP.computation_source_ids = sourceIds;
    end
    % Pace at most one payload unit per source into the Mode-2 queue.  The
    % accounting layer only admits a packet when its reservation precedes the
    % next safety beacon, so the single-packet queue cannot overflow.
    stationManagement = wilab_closed_loop_packet_accounting('pace', ...
        stationManagement, timeManagement, appParams, phyParams, ...
        timeManagement.timeNow);
    idLteHasPck = stationManagement.activeIDsCV2X( ...
        stationManagement.pckBuffer(stationManagement.activeIDsCV2X) >= 1);
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
    % The decode of this last replica happens at the end of this TTI.  Flag
    % the packets so the next TTI start can read their reception state
    % before pckReceived is reset for the following packet.
    if ~isempty(txIDlastTx)
        wilab_closed_loop_packet_accounting('mark_final', txIDlastTx);
    end
    % Start the next queued packet at its first replica/reservation attempt.
    stationManagement.pckNextAttempt(txIDlastTx) = 1;
    % reset of pckReceive and cumulativeSINR
    stationManagement.pckReceived(:,stationManagement.transmittingIDsCV2X(stationManagement.pckTxOccurring(stationManagement.transmittingIDsCV2X)==1)) = 0;
    sinrManagement.cumulativeSINR(:,stationManagement.transmittingIDsCV2X(stationManagement.pckTxOccurring(stationManagement.transmittingIDsCV2X)==1)) = 0;
end

