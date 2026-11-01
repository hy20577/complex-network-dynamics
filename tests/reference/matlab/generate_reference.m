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

%% fdr_bh and TPR_FPR (extracted unchanged from Mir_surrogate_data.m)
P = rd('in_fdr_pvals.txt');
for c = 1:size(P, 1)
    [h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(P(c, :), 0.05, 'pdep', 'no');
    wr(sprintf('ref_fdr_pdep_case%d.txt', c), [double(h); adj_p; crit_p * ones(1, 20); adj_ci_cvrg * ones(1, 20)]);
    [h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(P(c, :), 0.05, 'dep', 'no');
    wr(sprintf('ref_fdr_dep_case%d.txt', c), [double(h); adj_p; crit_p * ones(1, 20); adj_ci_cvrg * ones(1, 20)]);
end
[tpr, fpr] = TPR_FPR(rd('in_tpr_true.txt'), rd('in_tpr_inferred.txt'));
wr('ref_tpr_fpr.txt', [tpr, fpr]);

%% Mir_surrogate_data with stored surrogates (see shims_fixed_surrogates)
global FIXED_SURROGATES FIXED_SURROGATE_COUNTER
S5 = S(:, 1:5);
stacked = rd('in_surrogate_fixed.txt');
FIXED_SURROGATES = permute(reshape(stacked', 5, 1000, 20), [2 1 3]);
FIXED_SURROGATE_COUNTER = 0;
addpath(fullfile(here, 'shims_fixed_surrogates'));
[mir_all, adj_fdr, tp_fp] = Mir_surrogate_data(S5, 0.1, [500 1000], MatlabString("HT_mix_A_theta"), ...
                                               rd('in_surrogate_true_network.txt'), '', 1, 0.5);
rmpath(fullfile(here, 'shims_fixed_surrogates'));
for it = 1:2
    wr(sprintf('ref_surrogate_mir_overall_len%d.txt', it), mir_all(:, :, it));
    wr(sprintf('ref_surrogate_adjacency_len%d.txt', it), adj_fdr(:, :, it));
end
wr('ref_surrogate_tp_fp.txt', tp_fp);

%% double_normalised_MIR, lagged_double_MIR, best_case_TPR_FPR
A6 = rd('in_best_case_true.txt');
[nmir, max_rate] = double_normalised_MIR(S, A6);
wr('ref_double_normalised_mir.txt', nmir);
wr('ref_double_normalised_max_rate.txt', max_rate);

[lagged_mi, lagged_mir] = lagged_double_MIR(S(1:500, :), 2);
wr('ref_lagged_mi.txt', lagged_mi);
wr('ref_lagged_mir.txt', lagged_mir);

sims = {rd('in_best_case_similarity.txt'), rd('ref_mir_auto_Mir.txt')};
for c = 1:2
    [roc, best] = best_case_TPR_FPR(sims{c}, A6);
    wr(sprintf('ref_best_case_roc_case%d.txt', c), roc.data);
    wr(sprintf('ref_best_case_best_case%d.txt', c), best.data);
end

%% Synchronisation measures
ph = rd('in_sync_phases.txt');
wr('ref_global_sync.txt', global_sync(ph));
wr('ref_pairwise_sync.txt', pairwise_sync_of_phases(ph));
wr('ref_globalorder_map.txt', globalorder_map(S));
wr('ref_deviation_from_sync.txt', deviation_from_sync(S(:, 1), S(:, 2), 0.1 * (1:size(S, 1))'));

%% Model vector fields and Jacobians: with the tangent vectors set to the
% identity, an extended system returns [f; J(:)] (J column by column)
A12 = rd('in_logistic_adjacency.txt');
w12 = rd('in_model_w.txt');
th = rd('in_model_state_kuramoto.txt');
s3 = rd('in_model_state_3M.txt');
I12 = eye(12); I36 = eye(36);
wr('ref_kuramoto_ext.txt', Kuramoto_extended(0, [th(:); I12(:)], 2.0, A12, w12));
wr('ref_rossler_ext.txt', rossler_ext(0, [s3(:); I36(:)], A12, w12, 0.15, 0.2, 10, 0.3));
wr('ref_hr_ext.txt', HR_ext(0, [s3(:); I36(:)], A12, 0.2));

%% Coupled composite logistic maps (order 3): 5 transient + 5 recorded steps
[orbits, lya] = Composite_log_generator(A12, 0.1, x0, 4, 3, 5, 5, Inf, 5, 0);
wr('ref_composite_orbits.txt', orbits);
wr('ref_composite_lyapunov.txt', lya(end, :));

disp('Reference outputs written.');
