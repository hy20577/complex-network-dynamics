function SD_overall = HT_mix_A_theta(Dat, k, varargin)
% Test stand-in for HT_mix_A_theta: instead of random surrogates, returns the
% stored surrogates in global FIXED_SURROGATES (n_points x M x count) one by
% one, cut to the length of Dat. The Python test uses the same surrogates in
% the same order, so everything after surrogate generation can be compared
% exactly.
global FIXED_SURROGATES FIXED_SURROGATE_COUNTER
FIXED_SURROGATE_COUNTER = FIXED_SURROGATE_COUNTER + 1;
SD_overall = FIXED_SURROGATES(1:size(Dat, 1), :, FIXED_SURROGATE_COUNTER);
end
