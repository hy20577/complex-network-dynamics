function G = graph(s, t, varargin)
% Minimal stand-in for MATLAB's graph() so corr_decay.m runs in Octave.
% Like MATLAB, the graph has max(s, t) nodes and is undirected.
s = s(:); t = t(:);
G.s = s;
G.t = t;
G.n = max([s; t; 0]);
end
