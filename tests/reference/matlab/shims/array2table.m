function T = array2table(X, varargin)
% Minimal stand-in for MATLAB's array2table (Octave has no table type).
% Supports the 'VariableNames' option used by best_case_TPR_FPR.m.
names = {};
for k = 1:2:numel(varargin)
    if strcmpi(varargin{k}, 'VariableNames')
        names = varargin{k + 1};
    end
end
T = SimpleTable(X, names);
end
