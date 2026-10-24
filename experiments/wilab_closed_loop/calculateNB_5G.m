function [appParams,phyParams] = calculateNB_5G(appParams,phyParams)
% Isolated-experiment override of WiLab's calculateNB_5G.
%
% The vendor implementation derives the semi-persistent reservation period
% from appParams.allocationPeriod:
%
%     NbeaconsT = floor(allocationPeriod / Tslot_NR)
%
% That couples the controller cadence to the PHY reservation interval.  The
% corrected contract keeps the 20 ms controller clock but lets the PHY
% reserve resources every 5 ms, so the two clocks must be independent:
%
%     allocationPeriod   -> controller/application epoch (20 ms)
%     reservationInterval -> PSSCH semi-persistent reservation (5 ms)
%
% NbeaconsT remains the number of slots in one reservation period and is
% consumed by the Mode-2 resource-selection code exactly as before; only its
% source changes.  When reservationInterval is absent (legacy replays) the
% vendor behaviour is preserved.

% Find number of RBs per subchannel (or multiple)
phyParams.RBsBeaconSubchannel = phyParams.NsubchannelsBeacon*phyParams.sizeSubchannel;

% Find number of beacons in the frequency domain
if phyParams.BRoverlapAllowed
    appParams.NbeaconsF = phyParams.NsubchannelsFrequency - phyParams.NsubchannelsBeacon + 1;
else
    appParams.NbeaconsF = floor(appParams.RBsFrequencyV2V/(phyParams.RBsBeaconSubchannel));
end

% Find number of beacons in the time domain.  Use the explicit reservation
% interval when present; otherwise retain the legacy allocation-period link.
if isfield(appParams,'reservationInterval') && ~isempty(appParams.reservationInterval)
    appParams.NbeaconsT = floor(appParams.reservationInterval/phyParams.Tslot_NR);
else
    appParams.NbeaconsT = floor(appParams.allocationPeriod/phyParams.Tslot_NR);
end

if appParams.NbeaconsT < 1
    error('calculateNB_5G: reservationInterval yields NbeaconsT < 1');
end

end
