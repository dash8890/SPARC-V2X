function [F,X] = ecdf(values)
% Minimal toolbox-free empirical CDF compatible with WiLab output code.
values = values(:);
values = values(isfinite(values));
if isempty(values)
    F = zeros(0,1); X = zeros(0,1); return;
end
[X,~,idx] = unique(sort(values));
counts = accumarray(idx,1);
F = cumsum(counts) ./ numel(values);
end
