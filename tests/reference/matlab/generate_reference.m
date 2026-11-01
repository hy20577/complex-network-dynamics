% Runs the original MATLAB thesis functions in Octave on the inputs in ../data
% and writes their outputs there as reference values for the Python tests.
%
% Run from the project root:  octave --no-gui tests/reference/matlab/generate_reference.m

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'shims'));     % graph, distances, waitbar
addpath(fullfile(here, 'original'));  % unmodified thesis code
data = fullfile(here, '..', 'data');

rd = @(name) dlmread(fullfile(data, name), ',');
wr = @(name, X) dlmwrite(fullfile(data, name), X, 'delimiter', ',', 'precision', '%.17g');
warning('off', 'all');

%% location
for name = {'random', 'gridlines'}
    X = rd(sprintf('in_location_%s.txt', name{1}));
    for N = [4 7]
        [coords, ppb] = location(X, N);
        wr(sprintf('ref_location_%s_N%d_coords.txt', name{1}, N), coords);
        wr(sprintf('ref_location_%s_N%d_ppb.txt', name{1}, N), ppb);
    end
end

%% Grid, corr_decay (one pair at a time, as MIR.m uses it) and MIR
S = rd('in_mir_series.txt');
[N_min, N_max] = Grid(S);
wr('ref_grid.txt', [N_min, N_max]);

M = size(S, 2);
for N = [N_min N_max]
    T = zeros(M, M);
    for i = 1:M
        for j = 1:M
            T(i, j) = corr_decay([S(:, i) S(:, j)], N);
        end
    end
    wr(sprintf('ref_corr_decay_N%d.txt', N), T);
end

[Mir, Mi] = MIR(S);
wr('ref_mir_auto_Mir.txt', Mir);
wr('ref_mir_auto_Mi.txt', Mi);
[Mir, Mi] = MIR(S, [3 6]);
wr('ref_mir_3_6_Mir.txt', Mir);
wr('ref_mir_3_6_Mi.txt', Mi);

%% Coupled logistic maps (transient = time_window; std_threshold = Inf skips the adaptive part)
A = rd('in_logistic_adjacency.txt');
x0 = rd('in_logistic_x0.txt');
n_transient = 10;
[orbits, lya] = Log_generator(A, 0.1, x0, 4, 3000, n_transient, Inf, n_transient, 0);
wr('ref_logistic_orbits.txt', orbits(1:50, :));
wr('ref_logistic_lyapunov_final.txt', lya(end, :));

% Short run: 25 steps in total, before the chaotic trajectories separate.
% Lya_overtime averages over the transient too, so the Python run with no
% transient and 26 states covers exactly the same 25 Jacobians.
[~, lya] = Log_generator(A, 0.1, x0, 4, 15, 10, Inf, 10, 0);
wr('ref_logistic_lyapunov_25_steps.txt', lya(end, :));

% Isolated node: one transient step, then 10 recorded steps (11 Jacobians in all)
A_iso = rd('in_logistic_isolated_adjacency.txt');
x0_iso = rd('in_logistic_isolated_x0.txt');
[orbits, lya] = Log_generator(A_iso, 0.3, x0_iso, 4, 10, 1, Inf, 1, 0);
wr('ref_logistic_isolated_orbits.txt', orbits);
wr('ref_logistic_isolated_lyapunov.txt', lya(end, :));

%% Single Lorenz system with RK4 and Euler (no coupling: Adj = 0)
% RK4.m / Euler.m add the ODE output to a row state, so the column output of
% lorenz_ext is transposed here.
ode = @(t, X, varargin) reshape(lorenz_ext(t, X(:), varargin{:}), 1, []);
y0 = rd('in_lorenz_x0.txt');
args = {0, 0, 10, 28, 8/3};  % Adj, K, sigma, rho, beta
[t, orbits, lya] = RK4(ode, 0, y0, 20, 0.01, args{:});
wr('ref_rk4_lorenz_orbits.txt', orbits);
wr('ref_rk4_lorenz_lyapunov.txt', lya(end, :));
[t, orbits, lya] = Euler(ode, 0, y0, 5, 0.001, args{:});
wr('ref_euler_lorenz_orbits.txt', orbits);
wr('ref_euler_lorenz_lyapunov.txt', lya(end, :));

disp('Reference outputs written.');
