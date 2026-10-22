function varargout = wilab_event_replay(op, drawType, varargin)
% Record/replay stochastic MATLAB draws used by the isolated WiLab run.
% Values are exogenous draws; queue/resource evolution remains endogenous.
global WILAB_EVENT_REPLAY;
if nargin >= 1 && strcmp(op,'setup_record')
    file = drawType; WILAB_EVENT_REPLAY = struct('mode','record','file',file,'index',0,'types',{{}},'values',{{}}); return;
elseif nargin >= 1 && strcmp(op,'setup_replay')
    file = drawType; S = load(file,'types','values'); WILAB_EVENT_REPLAY = struct('mode','replay','file',file,'index',0,'types',{S.types},'values',{S.values}); return;
elseif nargin >= 1 && strcmp(op,'flush')
    if ~isempty(WILAB_EVENT_REPLAY) && strcmp(WILAB_EVENT_REPLAY.mode,'record')
        types=WILAB_EVENT_REPLAY.types; values=WILAB_EVENT_REPLAY.values; save(WILAB_EVENT_REPLAY.file,'types','values','-v7.3');
    end
    return;
elseif nargin >= 1 && strcmp(op,'draw')
    if isempty(WILAB_EVENT_REPLAY) || strcmp(WILAB_EVENT_REPLAY.mode,'off')
        [varargout{1:nargout}] = builtin(drawType,varargin{:}); return;
    end
    if strcmp(WILAB_EVENT_REPLAY.mode,'record')
        [varargout{1:nargout}] = builtin(drawType,varargin{:});
        WILAB_EVENT_REPLAY.index = WILAB_EVENT_REPLAY.index + 1;
        k=WILAB_EVENT_REPLAY.index; WILAB_EVENT_REPLAY.types{k}=drawType; WILAB_EVENT_REPLAY.values{k}=varargout{1};
    else
        WILAB_EVENT_REPLAY.index = WILAB_EVENT_REPLAY.index + 1; k=WILAB_EVENT_REPLAY.index;
        if k>numel(WILAB_EVENT_REPLAY.values) || ~strcmp(WILAB_EVENT_REPLAY.types{k},drawType)
            error('WiLabReplay:DrawMismatch','Replay draw mismatch at %d: expected %s, got %s',k, ternary(k<=numel(WILAB_EVENT_REPLAY.types),WILAB_EVENT_REPLAY.types{k},'EOF'),drawType);
        end
        varargout{1}=WILAB_EVENT_REPLAY.values{k};
    end
    return;
end
error('WiLabReplay:UnknownOperation','Unknown operation %s',op);
end

function y=ternary(c,a,b)
if c, y=a; else, y=b; end
end
