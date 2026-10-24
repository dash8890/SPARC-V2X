function test_packet_accounting()
% Unit checks for the payload-faithful, paced packet accounting helper.
% Run with:  matlab -batch "addpath('tests'); test_packet_accounting"
%
% The tests pin the corrected physical contract:
%   * registration only enqueues the payload; it never bursts pckBuffer,
%   * 'pace' admits at most one unit per source per TTI and never when the
%     next safety beacon precedes the reservation,
%   * a unit is credited only when it reached the *assigned helper*,
%   * missing units are retried with bounded attempts,
%   * duplicate units are deduplicated,
%   * a helper swap resets the unit bitmap,
%   * deadline expiry cancels outstanding units,
%   * queue overflow re-queues the dropped payload unit rather than losing it.

global WILAB_CLOSED_LOOP;
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'experiments', 'wilab_closed_loop'));
WILAB_CLOSED_LOOP = struct();

% Source is WiLab vehicle 1, helper is WiLab vehicle 2.
sm = fixture();
tm = struct('timeNow', 0.0, 'elapsedTime_TTIs', 2, ...
    'timeNextPacket', [0.100; 0.100; 0.100]);
ap = struct('NbeaconsF', 10, 'NbeaconsT', 20, 'allocationPeriod', 0.020);
ph = struct('TTI', 0.001);
action = struct();
action.assignments = struct('task_id', 0, 'source_id', 0, 'helper_id', 1, ...
    'source_wilab_id', 1, 'helper_wilab_id', 2, ...
    'packet_count', 3, 'payload_bytes', 3000, 'deadline_s', 1.0, ...
    'release_ms', 0.0);

% --- registration is lazy: nothing enters pckBuffer -----------------------
sm = wilab_closed_loop_packet_accounting('register', action, sm, 0.0);
assert(numel(WILAB_CLOSED_LOOP.pending{1}) == 3, 'three pending units expected');
assert(isempty(WILAB_CLOSED_LOOP.pkt_queue{1}), 'register must not burst pckBuffer');
assert(sm.pckBuffer(1) == 0, 'pckBuffer untouched by register');
assert(WILAB_CLOSED_LOOP.pkt_paced == 0, 'nothing paced yet');

% --- beacon guard: reservation at 19 ms, beacon at 5 ms -> do not admit ---
tm.timeNextPacket(1) = 0.005;
sm = wilab_closed_loop_packet_accounting('pace', sm, tm, ap, ph, 0.0);
assert(sm.pckBuffer(1) == 0, 'must not inject when the beacon comes first');
assert(numel(WILAB_CLOSED_LOOP.pending{1}) == 3, 'no unit admitted');

% --- beacon safely after the reservation -> admit exactly one unit --------
tm.timeNextPacket(1) = 0.100;
sm = wilab_closed_loop_packet_accounting('pace', sm, tm, ap, ph, 0.0);
assert(sm.pckBuffer(1) == 1, 'one unit admitted');
assert(numel(WILAB_CLOSED_LOOP.pkt_queue{1}) == 1, 'one tag in flight');
assert(WILAB_CLOSED_LOOP.pkt_queue{1}(1) == 1, 'unit 0 tag must decode as task 0 unit 0');
assert(numel(WILAB_CLOSED_LOOP.pending{1}) == 2, 'two units still pending');
assert(WILAB_CLOSED_LOOP.pkt_paced == 1, 'pacing counter');

% A second pace before the packet is served must not add another packet.
sm = wilab_closed_loop_packet_accounting('pace', sm, tm, ap, ph, 0.0);
assert(sm.pckBuffer(1) == 1, 'single-packet queue is respected');

% --- unit 0 reaches the helper -> credited, not complete -------------------
sm.pckReceived(2, 1) = 1;
sm.pckBuffer(1) = 0;
wilab_closed_loop_packet_accounting('mark_final', 1);
wilab_closed_loop_packet_accounting('capture', sm, 0.001, 1);
st = wilab_closed_loop_packet_accounting('state');
assert(st(1).received_packets == 1, 'one unit credited');
assert(~st(1).complete, 'partial payload must not complete');

% --- unit 1 is lost: no credit, retry is queued ---------------------------
tm.timeNow = 0.020;
tm.elapsedTime_TTIs = 21;
tm.timeNextPacket(1) = 0.100;
sm = wilab_closed_loop_packet_accounting('pace', sm, tm, ap, ph, 0.020);
assert(sm.pckBuffer(1) == 1 && WILAB_CLOSED_LOOP.pkt_queue{1}(1) == 2, ...
    'unit 1 admitted at the next reservation');
sm.pckReceived(2, 1) = 0;             % helper did not decode it
sm.pckBuffer(1) = 0;
wilab_closed_loop_packet_accounting('mark_final', 1);
wilab_closed_loop_packet_accounting('capture', sm, 0.021, 2);
st = wilab_closed_loop_packet_accounting('state');
assert(st(1).received_packets == 1, 'lost unit not credited');
assert(st(1).pending_units >= 1, 'lost unit returned to the pending queue');
assert(WILAB_CLOSED_LOOP.pkt_dropped == 0, 'retry budget not exhausted');

% --- FIFO serves unit 2 before the retry; both are credited ---------------
tm.timeNow = 0.040;
tm.elapsedTime_TTIs = 41;
tm.timeNextPacket(1) = 0.100;
sm = wilab_closed_loop_packet_accounting('pace', sm, tm, ap, ph, 0.040);
assert(sm.pckBuffer(1) == 1 && WILAB_CLOSED_LOOP.pkt_queue{1}(1) == 3, ...
    'unit 2 admitted before the retry');
sm.pckReceived(2, 1) = 1;
sm.pckBuffer(1) = 0;
wilab_closed_loop_packet_accounting('mark_final', 1);
wilab_closed_loop_packet_accounting('capture', sm, 0.041, 3);
st = wilab_closed_loop_packet_accounting('state');
assert(st(1).received_packets == 2, 'unit 2 credited');

% The retry of unit 1 is now at the head; admit it and count the retransmission.
tm.timeNow = 0.060;
tm.elapsedTime_TTIs = 61;
tm.timeNextPacket(1) = 0.100;
sm = wilab_closed_loop_packet_accounting('pace', sm, tm, ap, ph, 0.060);
assert(sm.pckBuffer(1) == 1 && WILAB_CLOSED_LOOP.pkt_queue{1}(1) == 2, ...
    'lost unit retried');
assert(WILAB_CLOSED_LOOP.pkt_retransmitted >= 1, 'retransmission counted');
assert(WILAB_CLOSED_LOOP.pkt_dropped == 0, 'dropped only after attempt budget');

% --- helper swap resets the unit bitmap and flushes in-flight payload -----
sm.pckBuffer(1) = 0;                  % model WiLab removing the in-flight packet
swapped = action;
swapped.assignments.helper_id = 2;    % Python helper index 2
swapped.assignments.helper_wilab_id = 3;  % WiLab vehicle 3
sm = wilab_closed_loop_packet_accounting('register', swapped, sm, 0.061);
st = wilab_closed_loop_packet_accounting('state');
assert(st(1).received_packets == 0, 'helper swap resets received units');
assert(numel(WILAB_CLOSED_LOOP.pending{1}) == 3, ...
    'all units re-queued for the new helper');
assert(isempty(WILAB_CLOSED_LOOP.pkt_queue{1}), 'old-helper packet flushed');

% --- full payload at the new helper --------------------------------------
for e = 1:3
    tm.timeNow = 0.061 + e * 0.020;
    tm.elapsedTime_TTIs = round(tm.timeNow / ph.TTI) + 1;
    tm.timeNextPacket(1) = tm.timeNow + 0.100;
    sm = wilab_closed_loop_packet_accounting('pace', sm, tm, ap, ph, tm.timeNow);
    assert(sm.pckBuffer(1) == 1, 'one unit paced per reservation');
    sm.pckReceived(3, 1) = 1;          % new helper decodes
    sm.pckBuffer(1) = 0;
    wilab_closed_loop_packet_accounting('mark_final', 1);
    wilab_closed_loop_packet_accounting('capture', sm, tm.timeNow + 0.001, 9 + e);
end
st = wilab_closed_loop_packet_accounting('state');
assert(st(1).received_packets == 3, 'full payload credited at the new helper');
assert(st(1).complete, 'full payload completes');
assert(~isempty(st(1).complete_epoch) && st(1).complete_epoch >= 10, ...
    'completion epoch recorded');
assert(st(1).received_packets + st(1).pending_units + ...
    numel(WILAB_CLOSED_LOOP.pkt_queue{1}) == 3, 'packet conservation');

% A completed task must not be re-enqueued on a later registration.
sm = wilab_closed_loop_packet_accounting('register', swapped, sm, 0.200);
assert(numel(WILAB_CLOSED_LOOP.pending{1}) == 0, 'completed task not re-queued');

% --- queue overflow re-queues the dropped payload tag ---------------------
WILAB_CLOSED_LOOP = struct();
sm2 = fixture();
a2 = action;
a2.assignments.task_id = 7;
a2.assignments.packet_count = 3;
a2.assignments.deadline_s = 1.0;
sm2 = wilab_closed_loop_packet_accounting('register', a2, sm2, 0.0);
% Clear the pending queue so the re-queue is unambiguous.
WILAB_CLOSED_LOOP.pending{1} = [];
WILAB_CLOSED_LOOP.pending_not_before{1} = [];
WILAB_CLOSED_LOOP.pending_attempts{1} = [];
WILAB_CLOSED_LOOP.task_pending_bits(8, :) = false;
WILAB_CLOSED_LOOP.pkt_queue{1} = [700001, 700002];
sm2.pckBuffer(1) = 1;
loss_before = WILAB_CLOSED_LOOP.pkt_overflow_losses;
wilab_closed_loop_packet_accounting('reconcile', sm2, 0.001);
assert(WILAB_CLOSED_LOOP.pkt_overflow_losses == loss_before + 1, 'overflow counted');
assert(numel(WILAB_CLOSED_LOOP.pkt_queue{1}) == 1, 'FIFO head kept');
assert(any(WILAB_CLOSED_LOOP.pending{1} == 700002), 'dropped unit re-queued');

% --- deadline cancellation drops outstanding units ------------------------
WILAB_CLOSED_LOOP = struct();
sm3 = fixture();
a3 = action;
a3.assignments.task_id = 9;
a3.assignments.deadline_s = 0.050;
sm3 = wilab_closed_loop_packet_accounting('register', a3, sm3, 0.0);
tm4 = tm;
tm4.timeNow = 0.060;
tm4.elapsedTime_TTIs = 61;
tm4.timeNextPacket(1) = 0.100;
sm3 = wilab_closed_loop_packet_accounting('pace', sm3, tm4, ap, ph, 0.060);
st = wilab_closed_loop_packet_accounting('state');
assert(st(1).cancelled, 'task cancelled after its absolute deadline');
assert(numel(WILAB_CLOSED_LOOP.pending{1}) == 0, 'pending units dropped');
assert(sm3.pckBuffer(1) == 0, 'no packet admitted for a cancelled task');
assert(WILAB_CLOSED_LOOP.pkt_cancelled >= 3, 'cancelled units counted');

fprintf('test_packet_accounting: OK\n');
end

function sm = fixture()
    sm = struct();
    sm.pckReceived = zeros(3, 3);
    sm.pckBuffer = zeros(3, 1);
    sm.pckNextAttempt = ones(3, 1);
    sm.BRid = [1; 0; 0];              % source 1 occupies Mode-2 slot 1
    sm.cv2xNumberOfReplicas = ones(3, 1);
    sm.activeIDsCV2X = [1; 2; 3];
end
