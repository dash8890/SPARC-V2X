function [cbr,p95DelayMs,cbrValid,delayValid,prr,prrValid,nReassign,windowPrr,windowPrrValid] = wilab_closed_loop_radio_metrics(stationManagement,outputValues,outParams)
% Extract controller-facing radio metrics without treating WiLab sentinels
% as measurements.  cbrCV2Xvalues uses -1 before a sensing interval exists.
cbr = 0; cbrValid = false;
prr = 0; prrValid = false;
nReassign = 0;
windowPrr = 1.0; windowPrrValid = false;

global WILAB_CLOSED_LOOP;
if isempty(WILAB_CLOSED_LOOP), WILAB_CLOSED_LOOP = struct(); end
if ~isfield(WILAB_CLOSED_LOOP,'prevOk'), WILAB_CLOSED_LOOP.prevOk = 0; end
if ~isfield(WILAB_CLOSED_LOOP,'prevErr'), WILAB_CLOSED_LOOP.prevErr = 0; end

if isfield(outputValues,'NcorrectlyTxBeaconsTOT') && isfield(outputValues,'NerrorsTOT')
    ok = sum(double(outputValues.NcorrectlyTxBeaconsTOT(:)));
    err = sum(double(outputValues.NerrorsTOT(:)));
    if ok + err > 0
        prr = ok / (ok + err);
        prrValid = true;
    end
    dOk = ok - WILAB_CLOSED_LOOP.prevOk;
    dErr = err - WILAB_CLOSED_LOOP.prevErr;
    if dOk + dErr > 0
        windowPrr = dOk / (dOk + dErr);
        windowPrrValid = true;
        WILAB_CLOSED_LOOP.prevOk = ok;
        WILAB_CLOSED_LOOP.prevErr = err;
        WILAB_CLOSED_LOOP.lastWindowPrr = windowPrr;
    elseif isfield(WILAB_CLOSED_LOOP, 'lastWindowPrr')
        windowPrr = WILAB_CLOSED_LOOP.lastWindowPrr;
        windowPrrValid = true;
    else
        windowPrr = 1.0;
        windowPrrValid = true;
    end
end
if isfield(stationManagement,'cbrCV2Xvalues') && ~isempty(stationManagement.cbrCV2Xvalues)
    vals = stationManagement.cbrCV2Xvalues;
    % WiLab stores CBR history by sensing interval.  The newest column can
    % remain the -1 startup sentinel even after earlier columns are valid;
    % select the most recent valid sample per vehicle rather than reading a
    % fixed terminal column.
    if ndims(vals) > 1
        valid = isfinite(vals) & vals >= 0 & vals <= 1;
        latest = nan(size(vals,1),1);
        for i = 1:size(vals,1)
            idx = find(valid(i,:),1,'last');
            if ~isempty(idx), latest(i) = vals(i,idx); end
        end
        vals = latest;
    end
    vals = vals(isfinite(vals) & vals >= 0 & vals <= 1);
    if ~isempty(vals)
        cbr = mean(vals);
        cbrValid = true;
    end
end

% pckReceived records attempted decodes (negative) and successful decodes
% (positive) for the current packet generation window.
if ~prrValid && isfield(stationManagement,'pckReceived') && ~isempty(stationManagement.pckReceived)
    rx = stationManagement.pckReceived;
    attempted = sum(rx(:) ~= 0);
    if attempted > 0
        prr = sum(rx(:) > 0) / attempted;
        prrValid = true;
    end
end

% WiLab stores packet-delay observations as a histogram.  Convert the
% accumulated histogram to a 95th-percentile delay in milliseconds.
p95DelayMs = 0; delayValid = false;
if isfield(outputValues,'packetDelayCounterCV2X') && ...
        ~isempty(outputValues.packetDelayCounterCV2X)
    h = outputValues.packetDelayCounterCV2X;
    nBins = size(h,3);
    counts = zeros(nBins,1);
    for k = 1:nBins
        counts(k) = sum(h(:,:,k,:), 'all');
    end
    total = sum(counts);
    if total > 0
        cdf = cumsum(counts) / total;
        k95 = find(cdf >= 0.95,1,'first');
        if ~isempty(k95)
            p95DelayMs = (k95-1) * outParams.delayResolution * 1000;
            delayValid = true;
        end
    end
end
if isfield(outputValues,'NreassignCV2X') && ~isempty(outputValues.NreassignCV2X)
    nReassign = double(outputValues.NreassignCV2X);
end
end
