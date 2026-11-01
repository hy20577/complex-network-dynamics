function y = round(x, n)
% Adds MATLAB's round(x, n) (n decimal digits), used in the progress message
% of Mir_surrogate_data.m. round(x) is passed to Octave's own round.
if nargin < 2
    y = builtin('round', x);
else
    y = builtin('round', x * 10^n) / 10^n;
end
end
