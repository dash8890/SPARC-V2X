function x = exprnd(mu,varargin)
% Minimal toolbox-free exponential sampler for the isolated WiLab bridge.
% Matches MATLAB's exprnd(mu) for the scalar use in WiLabV2Xsim.
if nargin < 1 || ~isscalar(mu) || ~isfinite(mu) || mu < 0
    error('exprnd:InvalidMean','mu must be a finite nonnegative scalar');
end
if nargin == 1
    x = -mu .* log(max(rand(), realmin));
else
    sz = varargin{1};
    x = -mu .* log(max(rand(sz), realmin));
end
end
