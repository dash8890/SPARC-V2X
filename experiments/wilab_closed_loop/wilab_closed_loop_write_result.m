function wilab_closed_loop_write_result(bridgeDir, episode, epoch, action, stationManagement, previousPckReceived, outputValues, outParams)
% Persist the interval outcome consumed by the Python application layer.
result.delivered_task_ids = [];
result.allowed_sources = [];
result.delta_packets = 0;
result.computation_packets_offered = 0;
result.computation_bytes_offered = 0;
result.wilab_enqueued_packets = 0;
result.wilab_enqueued_bytes = 0;
result.wilab_scheduled_attempts = 0;
result.wilab_pending_packets = 0;
result.rx_positive = 0;
result.pck_buffer_sum = 0;
result.safety_rx_cumulative = 0;
result.safety_error_cumulative = 0;
result.safety_prr_cumulative = 0;
result.safety_p95_delay_cumulative_ms = 0;
result.safety_delay_hist_cumulative = [];
result.delay_resolution_ms = 1.0;
result.radio_events = struct([]);
if isfield(action,'allowed_sources')
    result.allowed_sources = double(action.allowed_sources(:))';
end
if isfield(action,'computation_packets_offered')
    result.computation_packets_offered = double(action.computation_packets_offered);
end
if isfield(action,'computation_bytes_offered')
    result.computation_bytes_offered = double(action.computation_bytes_offered);
end
global WILAB_CLOSED_LOOP;
if ~isempty(WILAB_CLOSED_LOOP)
    if isfield(WILAB_CLOSED_LOOP,'instrument_enqueued_packets'), result.wilab_enqueued_packets = WILAB_CLOSED_LOOP.instrument_enqueued_packets; end
    if isfield(WILAB_CLOSED_LOOP,'instrument_enqueued_bytes'), result.wilab_enqueued_bytes = WILAB_CLOSED_LOOP.instrument_enqueued_bytes; end
    if isfield(WILAB_CLOSED_LOOP,'instrument_scheduled_attempts'), result.wilab_scheduled_attempts = WILAB_CLOSED_LOOP.instrument_scheduled_attempts; end
    if isfield(WILAB_CLOSED_LOOP,'instrument_pending_packets'), result.wilab_pending_packets = WILAB_CLOSED_LOOP.instrument_pending_packets; end
    if isfield(WILAB_CLOSED_LOOP,'radio_events'), result.radio_events = WILAB_CLOSED_LOOP.radio_events; end
    WILAB_CLOSED_LOOP.radio_events = struct([]);
end
if ~isempty(previousPckReceived) && isfield(stationManagement,'pckReceived')
    delta = sum(stationManagement.pckReceived,1) - sum(previousPckReceived,1);
    result.delta_packets = sum(delta(:));
    result.rx_packet_delta = result.delta_packets;
    result.rx_positive = sum(stationManagement.pckReceived(:) > 0);
    for s = result.allowed_sources
        ix = round(s) + 1; % Python task source IDs are zero based.
        % WiLab resets pckReceived when a new packet starts, so a simple
        % frame-to-frame sum can be zero even though the current packet was
        % successfully decoded.  Treat a positive receive entry for an
        % admitted source as the authoritative per-interval delivery event;
        % the Python task layer de-duplicates completed task IDs.
        if ix >= 1 && ix <= size(stationManagement.pckReceived,2) && ...
                any(stationManagement.pckReceived(:,ix) > 0)
            result.delivered_task_ids(end+1) = round(s); %#ok<AGROW>
        end
    end
    result.delta_packets = numel(result.delivered_task_ids);
end
if isfield(stationManagement,'pckBuffer')
    result.pck_buffer_sum = sum(stationManagement.pckBuffer(:));
end
if nargin >= 7 && ~isempty(outputValues)
    if nargin >= 8 && isfield(outParams,'delayResolution')
        result.delay_resolution_ms = double(outParams.delayResolution) * 1000;
    end
    if isfield(outputValues,'NcorrectlyTxBeaconsTOT') && isfield(outputValues,'NerrorsTOT')
        result.safety_rx_cumulative = sum(double(outputValues.NcorrectlyTxBeaconsTOT(:)));
        result.safety_error_cumulative = sum(double(outputValues.NerrorsTOT(:)));
        denom = result.safety_rx_cumulative + result.safety_error_cumulative;
        if denom > 0, result.safety_prr_cumulative = result.safety_rx_cumulative / denom; end
    end
    if nargin >= 8 && isfield(outputValues,'packetDelayCounterCV2X') && ~isempty(outputValues.packetDelayCounterCV2X)
        h=outputValues.packetDelayCounterCV2X; nb=size(h,3); counts=zeros(nb,1);
        for kk=1:nb, counts(kk)=sum(h(:,:,kk,:), 'all'); end
        total=sum(counts);
        result.safety_delay_hist_cumulative = counts(:)';
        if total>0
            cdf=cumsum(counts)/total; k95=find(cdf>=0.95,1,'first');
            if ~isempty(k95), result.safety_p95_delay_cumulative_ms=(k95-1)*outParams.delayResolution*1000; end
        end
    end
    % Fig.7d: PRR vs distance from WiLab distanceDetailsCounterCV2X
    % layout: [distance, #correct, #errors, #blocked, #neighbors]
    if isfield(outputValues,'distanceDetailsCounterCV2X') && ~isempty(outputValues.distanceDetailsCounterCV2X)
        D = outputValues.distanceDetailsCounterCV2X;
        % Aggregate over channels/packet-types: sum dims 1..end-2, keep distance bins x stats
        while ndims(D) > 2
            D = sum(D, 1);
            D = squeeze(D);
        end
        nb = size(D, 1);
        rows = struct('distance', {}, 'correct', {}, 'errors', {}, 'neighbors', {}, 'prr', {});
        for ii = 1:nb
            dist = double(D(ii, 1));
            ok = double(D(ii, min(2, size(D,2))));
            err = 0; neigh = 0;
            if size(D,2) >= 3, err = double(D(ii, 3)); end
            if size(D,2) >= 5, neigh = double(D(ii, 5)); end
            denom = ok + err;
            if denom <= 0 && neigh > 0, denom = neigh; end
            prr = 0;
            if denom > 0, prr = ok / denom; end
            rows(end+1) = struct('distance', dist, 'correct', ok, 'errors', err, ...
                'neighbors', neigh, 'prr', prr); %#ok<AGROW>
        end
        result.prr_vs_distance = rows;
    end
end
result.episode = episode;
result.epoch = epoch;
result.cbr = 0;
result.prr = 1;
result.p95_delay_ms = 0;
fname = fullfile(bridgeDir, sprintf('result_%06d_%06d.json',episode,epoch));
fid = fopen(fname,'w'); fwrite(fid,jsonencode(result),'char'); fclose(fid);
end
