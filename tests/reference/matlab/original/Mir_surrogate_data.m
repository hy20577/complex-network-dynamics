
function [MIR_overall, adj_MIR_fdr, TP_FP_Rate] = Mir_surrogate_data(All_Dat, p_sl, time_range, type_of_surr, original_network, label, varargin)

%%% COMPULSORY INPUTs

% All_Dat: Data to use computation of MIR after removing transient period.
% type_of_surr: "twnsd", "randsd", "corr_randsd", 'IAAFT2', 'HT_phase_shuffle', 'HT_phase_rand' 

%%% OPTIONAL INPUTs 

% p_sl: confidence interval for hypothesis testing (default: 0.01).
% desired_corr: When the surrogate data is generated as correlated between
        % pairs, you need to define desired correlation otherwise it is
        % defined as 0.5 as default.
        
% OUTPUT

    % MIR_overall : First row gives MIR value for original data. Next
           % 1/p_sl row gives the MIR values for surrogate data.
    % adj_MIR_fdr : gives the adjacency matrix (symetric) after fdr
                % process.
    % TP_FP_Rate : nx3 matrix. 1st column -- Time, 2nd -- TPR, 3rd -- FPR.
    
time_start = tic;

if ~exist('label', 'var')
label = '';
end

if nargin == 6
     pc1 = 1;
     pc2 = 0.5;
elseif nargin == 7
   pc1 = varargin{1};
   pc2 = 0.5;
elseif nargin ==8
    pc1 = varargin{1};
    pc2 = varargin{2};
end


if ~exist("time_range",'var')
time_range = logspace(3e2,size(All_Dat,1),100);
end


if time_range(end) > size(All_Dat,1)
    time_range(end) = size(All_Dat,1);
    warning('the last element of time_range should be less than number of observations in original data, assigned the biggest possible value automatically')
end

if ~exist("p_sl",'var')
    p_sl = 0.01;
end

if ~exist("desired_corr", "var")
    desired_corr = 0.5;
end

if type_of_surr == "HT_mix_A_theta"
         if nargin == 6
             pc1 = 1;
             pc2 = 0.5;
         elseif nargin == 7
           pc1 = varargin{1};
           pc2 = 0.5;
         elseif nargin ==8
            pc1 = varargin{1};
            pc2 = varargin{2};
         end

end


% create folder for outputs
% currentpath = pwd;
% name_of_folder = sprintf('%s/%s p_sl = %.2f %s', currentpath, type_of_surr, p_sl, label); 
% if exist(sprintf('%s/%s p_sl = %.2f %s', currentpath, type_of_surr, p_sl, label) , 'dir')
%     rmdir(name_of_folder, 's')
%     mkdir(name_of_folder)
% else
%     mkdir(name_of_folder)
% end
%%%%%%%%%%%%%%%%%%%%


nVar = size(All_Dat,2);
Nsd = 1/p_sl;   % Number of surrogate data
MIR_overall = zeros(1+Nsd,nVar*(nVar-1)/2, length(time_range));
Inferred_Adj_MIR = zeros(nVar, nVar,length(time_range));
adj_MIR_fdr = zeros(nVar,nVar, length(time_range));

TP_FP_Rate = zeros(length(time_range),3);
TP_FP_Rate(:,1) = time_range;

Dur = zeros(length(time_range),1);


for ITER=1:length(time_range)

    Dat = All_Dat(1:time_range(ITER),:);

    [MIR_orig, ~] = MIR(Dat);

    niter = 0;

    for n = 1:nVar
        for m = n+1:nVar
          niter = niter + 1;
          MIR_overall(1,niter,ITER) = MIR_orig(n,m);
        end
    end

    % Producing normal surrogate data and their MI values.
MIR_surrogate = zeros(Nsd, nVar*(nVar-1)/2);

parfor i=1:Nsd %%parfor
     
     surr_dat = zeros(size(Dat));   
     if type_of_surr == "twnsd"
        surr_dat = 	phaseran(Dat,1);
        % phaserun produces twin surrogate data
        % to test the null hypotheses
     elseif  type_of_surr == "randsd"
        surr_dat = rand(length(Dat), nVar);

     elseif type_of_surr == "corr_randsd"
        mu = 0;
        sigma = 1;
        M = mu + sigma*randn(1000,nVar);
        R = ones(nVar,nVar)*desired_corr;
        
        for k=1:nVar
            R(k,k) = 1;
        end
        L = chol(R);
        surr_dat = M*L;
    
     elseif type_of_surr == "IAAFT2"
    
        surr_dat = zeros(size(Dat,1), size(Dat,2));
        for ik=1:size(Dat,2)
            [surr,~]=surrogate(Dat(:,ik), 1, 'IAAFT2', 0, 40);  % Fourier Surrogate Data
            surr_dat(:,ik) = surr';
        end
% 
%      elseif type_of_surr == "HT_phase_shuffle"
% 
%          [surr_dat ,~] = HT_phase_rand_shuff(Dat, "shuffle", 1);
%      elseif type_of_surr == "HT_phase_rand"
%          [surr_dat ,~] = HT_phase_rand_shuff(Dat, "random", 1);
% 
%      elseif type_of_surr == "HT_A_rand"
%         surr_dat = HT_A_adjust(Dat,"rand", 1)
%      elseif type_of_surr == "HT_A_blockshuffled"
%          surr_dat =  HT_A_adjust(Dat,"blockshuffled", 1, 2e4);

     elseif type_of_surr == "HT_mix_A_theta"         
            surr_dat = HT_mix_A_theta(Dat, 1, pc1, pc2);
     end

       [MIR_surr, ~] = MIR(surr_dat);
       niter=0;
       MIR_array = zeros(1,nVar*(nVar-1)/2);

       for n = 1:nVar
            for m = n+1:nVar
              niter = niter+1;
              MIR_array(niter) = MIR_surr(n,m);
            end
       end  
        MIR_surrogate(i,:) = MIR_array;

end
    MIR_overall(2:end,:,ITER) = MIR_surrogate;

    % checking number of MI from surrogate data which is bigger than or equal to original MI

    c_xy_MIR = zeros(1, nVar*(nVar-1)/2);
    
    for i=1:nVar*(nVar-1)/2
        sum1 = 0;
       for j=2:Nsd+1

          if MIR_overall(j,i,ITER) > MIR_overall(1,i,ITER) || MIR_overall(j,i,ITER) == MIR_overall(1,i,ITER)
              sum1 = sum1 + 1;
          end

          c_xy_MIR(i) = sum1;
       end
    end

    prob_MIR = c_xy_MIR / Nsd;

    iter = 0;

    for i=1:nVar
         for j= i+1:nVar
          iter = iter+1;

          if prob_MIR(iter) < p_sl
                Inferred_Adj_MIR(i,j,ITER) = 1;
          end

         end
    end

    Inferred_Adj_MIR(:,:,ITER) = Inferred_Adj_MIR(:,:,ITER) + Inferred_Adj_MIR(:,:,ITER)';
   
% Calculation of False Discovery Rate (FDR)

     [adj, ~, ~, ~]= fdr_bh(prob_MIR,p_sl/100,'pdep','no');
     
     counter = 0;
     for i=1:nVar
        for j=i+1:nVar
          counter = counter+1;
          adj_MIR_fdr(i,j,ITER) = adj(counter);
        end
     end
  
  adj_MIR_fdr(:,:,ITER) = adj_MIR_fdr(:,:,ITER) + adj_MIR_fdr(:,:,ITER)'; 
  [TP_Rate,FP_Rate] = TPR_FPR(original_network,adj_MIR_fdr(:,:,ITER));
  TP_FP_Rate(ITER,2) = TP_Rate;
  TP_FP_Rate(ITER,3) = FP_Rate; 

%    if ITER > 3
%        if (TP_FP_Rate(ITER-2,2) == 1) && (TP_FP_Rate(ITER-1,2) == 1) && (TP_FP_Rate(ITER,2) == 1)
%            if (TP_FP_Rate(ITER-2,3) == 0) && (TP_FP_Rate(ITER-1,3) == 0) && (TP_FP_Rate(ITER,3) == 0)
%             break
%            end
%        end
%    end
%    
   %%%%%%%%% PLOTS and saving the MATRIX  %%%%%%%%

% Original Network vs. Inferred Network


% figure('Visible',"off")
% tiledlayout(1,2);
% nexttile
% plot(graph(original_network),"Layout","circle")
% 
% title('Original Network','Interpreter','latex',"FontSize",20) 
% 
% nexttile
% plot(graph(adj_MIR_fdr(:,:,ITER)),"Layout","circle")
% title('Inferred Network', 'Interpreter','latex',"FontSize",20)
% saveas(gcf,sprintf('%s/OriginalvsInferredNetwork %s and %g.png',name_of_folder, label, p_sl));

% TPR and FPR 
   
% figure('Visible',"off")
% t = tiledlayout(1,3);
% nexttile
% plot(TP_FP_Rate(:,3),TP_FP_Rate(:,2), 'o--','MarkerFaceColor',[1 .6 .6])
% xlabel('FP')
% ylabel('TP')
% title('TP vs FP', 'Interpreter','latex',"FontSize",20)
% xlim([0,1])
% ylim([0,1])
% refline(1,0)
% 
% nexttile
% loglog(TP_FP_Rate(:,1),TP_FP_Rate(:,2),'o--', 'MarkerFaceColor',[1 .6 .6])
% xlabel('Length of Time Series')
% ylabel('TP')
% title('TP vs Length', 'Interpreter','latex',"FontSize",20)
% ylim([0,1])
% 
% nexttile
% loglog(TP_FP_Rate(:,1),TP_FP_Rate(:,3),'s--','MarkerFaceColor',[1 .6 .6])
% xlabel('Length of Time Series')
% ylabel('FP')
% title('FP vs Length', 'Interpreter','latex',"FontSize",20)
% ylim([0,1])
% 
% saveas(gcf, sprintf('%s/%s & p_sl = %g.png',name_of_folder, label, p_sl));


% Saving the output   
% writematrix(MIR_overall, sprintf('%s/%s & p_sl= %g MIR_overall.txt',name_of_folder,  label, p_sl))
% writematrix(adj_MIR_fdr, sprintf('%s/%s & p_sl= %g adj_MIR_fdr.txt',name_of_folder,  label, p_sl))
% writematrix(TP_FP_Rate, sprintf('%s/%s & p_sl= %g TP_FP_Rate.txt',name_of_folder, label, p_sl))

%%%% For estimation
% of remaining time
    Dur(ITER) = toc(time_start);

%%% Try exponential regression but didn't work well.

%     exp_ft = fit((1:ITER)', Dur(1:ITER), 'exp2');
%     remaining_time_seconds = exp_ft(length(time_range))-Dur(ITER);

%%% Let's consider the data length in each iteration.

rate_of_completed = sum(time_range(1:ITER))/sum(time_range);
remaining_time_seconds = ((Dur(ITER))/rate_of_completed)-(Dur(ITER));
 
fprintf("Completed: %s %% \n ETA: %s (in DD:HH:MM:SS.MS)\n", num2str(round(rate_of_completed*100,1)), datestr(remaining_time_seconds/(24*60*60),'DD:HH:MM:SS.FFF') )

% fprintf('Progression : %s %% \n', num2str(round(ITER*100/length(time_range))))

end

fprintf('Elapsed time (function:Mir_surrogate_data) for %s and p_sl: %g = %s (in DD:HH:MM:SS.MS)\n',label,p_sl,datestr(toc(time_start)/(24*60*60),'DD:HH:MM:SS.FFF')) %, '%s iteration number\n', num2str(i), ' out of ', num2str(k));

end


%%  FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% MIR to compute mean of MIR between pairs over grid size and its sub-functions

function [Mir, Mi] = MIR(Dat, nodes)

%%% The function computes MIR values (without normalisation as in Bianco et.
% al. 2016) and chose the maximum partition size N which satisfies the 
% unbiased probability calculation in eq.17 in the same paper. 

%%% INPUT:

% Time-series data for different nodes

% OPTIONAL INPUT

% nodes = [p,q] to produce evolution of the MI and MIR over partitions.

%%% OUTPUT

% Mir : Mutual Information Rate

% Mi : Mutual Information

M = length(Dat(1,:));
[N_min, N_max] = Grid(Dat);

Is = zeros(M,M);  %% Mutual Information between pairs
Ic = zeros(M,M);  %% MIR between pairs gaining by dividing MI by t.
Mi_MIR_over_partitions = zeros(2,length(N_min:N_max));

for N = N_min:N_max
    
I_s = zeros(M,M);  %% Mutual Information between pairs
I_c = zeros(M,M);  %% MIR between pairs gaining by dividing MI by t.
    
      for i=1:M-1  %par
        for j=(i+1):M
        node1 = Dat(:,i);
        node2 = Dat(:,j);
        node12 = [node1, node2];
        [coordinates, ppb] = location(node12,N);        
        t = corr_decay(node12, N); %% correlation decay time for each pair.
        t = t(t~=0);
       
        MI=0;
               for m=1:N^2
               prob_node1 = sum(coordinates(:,2) == ceil(m/N)) / length(node1);
                 if mod(m,N) ~= 0
                    prob_node2 = sum(coordinates(:,3) == mod(m,N))/length(node1);
                 else 
                    prob_node2 = sum(coordinates(:,3) == N)/length(node1);
                 end
               joint_prob = ppb(m,2)/length(node1);
                         if joint_prob ~= 0  %%  p*log(p) -> 0 assumed as p ->0.
                         MI = MI + joint_prob * log(joint_prob/(prob_node1*prob_node2));
                         end             
               end
       
             I_s(i,j) =  MI;
             I_c(i,j) =  MI/t;
             
             
             if exist('nodes','var')        
                 if nodes(1) == i && nodes(2) == j     
                     Mi_MIR_over_partitions(1,N-N_min+1) = MI;
                     Mi_MIR_over_partitions(2,N-N_min+1) = MI/t;
                     
                 end
                 
             end
             
        end
      end
 Is = Is + I_s ;  
 Ic = Ic + I_c ; 
           
end
      
Mir = Ic/length(N_min:N_max);
Mi = Is/length(N_min:N_max);
 if exist('nodes','var')
     tiledlayout(1,2)
     nexttile
     loglog(N_min:N_max, Mi_MIR_over_partitions(1,:),'o--')
     title('Evolution of MI over partitions')
     nexttile
     loglog(N_min:N_max, Mi_MIR_over_partitions(2,:),'.-')
     title('Evolution of MIR over partitions')
 end

end

%%% Grid Function

function [N_min, N_max] = Grid(Dat)
% N_min: the smallest grid size
% N_max: the largest grid size satisfying the eq. 17. 
% Dat is the whole data n by M. 


[~, M] = size(Dat);
N_final = zeros(M*(M-1)/2,1);
r=0; 

for k=1:M-1
    for m=k+1:M
     N = 1;
     node1 = Dat(:,k);
     node2 = Dat(:,m);
    
     N_c = 0;
     mean_points = 1;
     
        while (mean_points >= N_c)
        N = N+1;
        [~, ppb] = location([node1, node2],N);        
        
        N_c = sum(ppb(:,2) ~= 0);
        mean_points = length(node1)/N^2;
        end
        r=r+1;
        N_final(r) = N-1;  % use this to create loop from N_min to N_max to consider all NxN partitions.
    end
end

N_min = round(0.1*mean(N_final))+1;
N_max = round(mean(N_final));
        if N_max <=2 
            N_max = N_max*2;
        end
        
        if N_min <=2 
           N_min = N_min*2;
        end
end

%%% corr_decay
% 
% function corr_decay = corr_decay(Dat,gridsize)
% % Correlation decay time for all pairs (columns) in given data.
% % Inputs 
% % Dat: time-series data.  gridsize: number of columns of rows, N.
% 
% M=length(Dat(1,:));
%    for i=1:M
%         for j=i+1:M
%             node1 = Dat(:,i); 
%             node2 = Dat(:,j);
%         
%             [coordinates, ~] = location([node1 node2],gridsize);        
%             DG = full(sparse(coordinates(1:end-1,4), coordinates(2:end,4), 1,gridsize^2,gridsize^2));
%             DG(DG ~= 0) =1;
%             DG = sparse(DG);
%             
%             % view(biograph(DG,[],'ShowArrows','off'));
%             
%             ShortestPath = graphallshortestpaths(DG);
%             ShortestPath(ShortestPath==Inf) = 0;
%             corr_decay = max(max(ShortestPath));
%         end
%    end
%             
% end

%%% location() 

function [coordinates, ppb] = location(Dat,grid)
% Dat is the npts by 2 matrix.  
% coordinates is a npts by 4 matrix.
% First column: Rank of the point
% Second Column: Row of the point in partition.
% Third Column: Column of the point in partition.
% Forth Column: Number of square which point lies.
% ppb: vector with N^2 length, which gives the number of points in cells.
 
npts = length(Dat(:,1));
coordinates = zeros(npts, 4);
ppb  = zeros(grid^2, 2);

min_v1 = min(Dat(:,1));
max_v1 = max(Dat(:,1));
Dat(:,1) = (Dat(:,1) - min_v1)/(max_v1 - min_v1);

min_v2 = min(Dat(:,2));
max_v2 = max(Dat(:,2));
Dat(:,2) = (Dat(:,2) - min_v2)/(max_v2 - min_v2);

Dat = Dat*grid;

Dat (Dat == 0) = 1e-15;
Dat(Dat == 1) = 1 - 1e-15;

    for i=1:npts
        coordinates(i,1) = i;
        coordinates(i,2) = ceil(Dat(i,1));
        coordinates(i,3) = ceil(Dat(i,2));
        coordinates(i,4) = coordinates(i,3) + (coordinates(i,2)-1)*grid;
    end

%%% Computation of ppb

nob = grid^2;
boxcounter = zeros(nob,1);


    for i=1:npts
    boxcounter(coordinates(i,4)) = boxcounter(coordinates(i,4)) + 1; 
    end
    
ppb(:,1) = (1:nob)';
ppb(:,2) = boxcounter;

end

%% 2. phaseran for twin surrogate data

function surrblk = phaseran(recblk,nsurr)

% Input data
% ----------
% recblk: is a 2D array. Row: time sample. Column: recording.
% An odd number of time samples (height) is expected. If that is not
% the case, recblock is reduced by 1 sample before the surrogate
% data is created.
% The class must be double and it must be nonsparse.
%
% nsurr: is the number of image block surrogates that you want to 
% generate.
% 
% Output data
% ---------------------
% surrblk: 3D multidimensional array image block with the surrogate
% datasets along the third dimension
% 
% Example 1
% ---------
%   x = randn(31,4);
%   x(:,4) = sum(x,2); % Create correlation in the data
%   r1 = corrcoef(x) 
%   surr = phaseran(x,10);
%   r2 = corrcoef(surr(:,:,1)) % Check that the correlation is preserved

%   Carlos Gias
%   Date: 21/08/2011

% Reference:
% Prichard, D., Theiler, J. Generating Surrogate Data for Time Series
% with Several Simultaneously Measured Variables (1994)
% Physical Review Letters, Vol 73, Number 7

% Get parameters
[nfrms,nts] = size(recblk);
if rem(nfrms,2)==0
    nfrms = nfrms-1;
    recblk=recblk(1:nfrms,:);
end
    
% Get parameters
len_ser = (nfrms-1)/2;
interv1 = 2:len_ser+1; 
interv2 = len_ser+2:nfrms;

% Fourier transform of the original dataset
fft_recblk = fft(recblk);

% Create the surrogate recording blocks one by one
surrblk = zeros(nfrms, nts, nsurr);
for k = 1:nsurr
   ph_rnd = rand([len_ser 1]);
   
   % Create the random phases for all the time series
   ph_interv1 = repmat(exp( 2*pi*1i*ph_rnd),1,nts);
   ph_interv2 = conj( flipud( ph_interv1));
   
   % Randomize all the time series simultaneously
   fft_recblk_surr = fft_recblk;
   fft_recblk_surr(interv1,:) = fft_recblk(interv1,:).*ph_interv1;
   fft_recblk_surr(interv2,:) = fft_recblk(interv2,:).*ph_interv2;
   
   % Inverse transform
   surrblk(:,:,k)= real(ifft(fft_recblk_surr));
end
end

%% 3. Roc Curve and TPR-FPR

function [TPR,FPR] = TPR_FPR(Orig_Network,Inferred_Network)
TP_SSM = 0;
TN_SSM = 0;
FP_SSM = 0;
FN_SSM = 0;
adj_SSM = Inferred_Network;
A = Orig_Network;
M = size(A,2);

for i=1:M
  for j=1:M

          if adj_SSM(i,j) ==1  && A(i,j)== 1
            TP_SSM = TP_SSM + 1;
          end
           if adj_SSM(i,j) == 1 && A(i,j) == 0  
              FP_SSM = FP_SSM + 1;
          end
          if  adj_SSM(i,j) == 0 && A(i,j) == 0  
              TN_SSM = TN_SSM + 1;
          end
          if  adj_SSM(i,j) == 0 && A(i,j) == 1  
              FN_SSM = FN_SSM + 1;
          end

   end
end

            TPR = TP_SSM/(TP_SSM+FN_SSM);
            FPR = FP_SSM / (FP_SSM + TN_SSM);

end

%% 4. False Discovery Rate (fdr procedure)

% fdr_bh() - Executes the Benjamini & Hochberg (1995) and the Benjamini &
%            Yekutieli (2001) procedure for controlling the false discovery 
%            rate (FDR) of a family of hypothesis tests. FDR is the expected
%            proportion of rejected hypotheses that are mistakenly rejected 
%            (i.e., the null hypothesis is actually true for those tests). 
%            FDR is a somewhat less conservative/more powerful method for 
%            correcting for multiple comparisons than procedures like Bonferroni
%            correction that provide strong control of the family-wise
%            error rate (i.e., the probability that one or more null
%            hypotheses are mistakenly rejected).
%
%            This function also returns the false coverage-statement rate 
%            (FCR)-adjusted selected confidence interval coverage (i.e.,
%            the coverage needed to construct multiple comparison corrected
%            confidence intervals that correspond to the FDR-adjusted p-values).
%
%
% Usage:
%  >> [h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(pvals,q,method,report);
%
% Required Input:
%   pvals - A vector or matrix (two dimensions or more) containing the
%           p-value of each individual test in a family of tests.
%
% Optional Inputs:
%   q       - The desired false discovery rate. {default: 0.05}
%   method  - ['pdep' or 'dep'] If 'pdep,' the original Bejnamini & Hochberg
%             FDR procedure is used, which is guaranteed to be accurate if
%             the individual tests are independent or positively dependent
%             (e.g., Gaussian variables that are positively correlated or
%             independent).  If 'dep,' the FDR procedure
%             described in Benjamini & Yekutieli (2001) that is guaranteed
%             to be accurate for any test dependency structure (e.g.,
%             Gaussian variables with any covariance matrix) is used. 'dep'
%             is always appropriate to use but is less powerful than 'pdep.'
%             {default: 'pdep'}
%   report  - ['yes' or 'no'] If 'yes', a brief summary of FDR results are
%             output to the MATLAB command line {default: 'no'}
%
%
% Outputs:
%   h       - A binary vector or matrix of the same size as the input "pvals."
%             If the ith element of h is 1, then the test that produced the 
%             ith p-value in pvals is significant (i.e., the null hypothesis
%             of the test is rejected).
%   crit_p  - All uncorrected p-values less than or equal to crit_p are 
%             significant (i.e., their null hypotheses are rejected).  If 
%             no p-values are significant, crit_p=0.
%   adj_ci_cvrg - The FCR-adjusted BH- or BY-selected 
%             confidence interval coverage. For any p-values that 
%             are significant after FDR adjustment, this gives you the
%             proportion of coverage (e.g., 0.99) you should use when generating
%             confidence intervals for those parameters. In other words,
%             this allows you to correct your confidence intervals for
%             multiple comparisons. You can NOT obtain confidence intervals 
%             for non-significant p-values. The adjusted confidence intervals
%             guarantee that the expected FCR is less than or equal to q
%             if using the appropriate FDR control algorithm for the  
%             dependency structure of your data (Benjamini & Yekutieli, 2005).
%             FCR (i.e., false coverage-statement rate) is the proportion 
%             of confidence intervals you construct
%             that miss the true value of the parameter. adj_ci=NaN if no
%             p-values are significant after adjustment.
%   adj_p   - All adjusted p-values less than or equal to q are significant
%             (i.e., their null hypotheses are rejected). Note, adjusted 
%             p-values can be greater than 1.
%
%
% References:
%   Benjamini, Y. & Hochberg, Y. (1995) Controlling the false discovery
%     rate: A practical and powerful approach to multiple testing. Journal
%     of the Royal Statistical Society, Series B (Methodological). 57(1),
%     289-300.
%
%   Benjamini, Y. & Yekutieli, D. (2001) The control of the false discovery
%     rate in multiple testing under dependency. The Annals of Statistics.
%     29(4), 1165-1188.
%
%   Benjamini, Y., & Yekutieli, D. (2005). False discovery rate?adjusted 
%     multiple confidence intervals for selected parameters. Journal of the 
%     American Statistical Association, 100(469), 71?81. doi:10.1198/016214504000001907
%
%
% Example:
%  nullVars=randn(12,15);
%  [~, p_null]=ttest(nullVars); %15 tests where the null hypothesis
%  %is true
%  effectVars=randn(12,5)+1;
%  [~, p_effect]=ttest(effectVars); %5 tests where the null
%  %hypothesis is false
%  [h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh([p_null p_effect],.05,'pdep','yes');
%  data=[nullVars effectVars];
%  fcr_adj_cis=NaN*zeros(2,20); %initialize confidence interval bounds to NaN
%  if ~isnan(adj_ci_cvrg),
%     sigIds=find(h);
%     fcr_adj_cis(:,sigIds)=tCIs(data(:,sigIds),adj_ci_cvrg); % tCIs.m is available on the
%     %Mathworks File Exchagne
%  end
%
%
% For a review of false discovery rate control and other contemporary
% techniques for correcting for multiple comparisons see:
%
%   Groppe, D.M., Urbach, T.P., & Kutas, M. (2011) Mass univariate analysis 
% of event-related brain potentials/fields I: A critical tutorial review. 
% Psychophysiology, 48(12) pp. 1711-1725, DOI: 10.1111/j.1469-8986.2011.01273.x 
% http://www.cogsci.ucsd.edu/~dgroppe/PUBLICATIONS/mass_uni_preprint1.pdf
%
%
% For a review of FCR-adjusted confidence intervals (CIs) and other techniques 
% for adjusting CIs for multiple comparisons see:
%
%   Groppe, D.M. (in press) Combating the scientific decline effect with 
% confidence (intervals). Psychophysiology.
% http://biorxiv.org/content/biorxiv/early/2015/12/10/034074.full.pdf
%
%
% Author:
% David M. Groppe
% Kutaslab
% Dept. of Cognitive Science
% University of California, San Diego
% March 24, 2010

%%%%%%%%%%%%%%%% REVISION LOG %%%%%%%%%%%%%%%%%
%
% 5/7/2010-Added FDR adjusted p-values
% 5/14/2013- D.H.J. Poot, Erasmus MC, improved run-time complexity
% 10/2015- Now returns FCR adjusted confidence intervals

function [h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(pvals,q,method,report)

if nargin<1
    error('You need to provide a vector or matrix of p-values.');
else
    if ~isempty(find(pvals<0,1))
        error('Some p-values are less than 0.');
    elseif ~isempty(find(pvals>1,1))
        error('Some p-values are greater than 1.');
    end
end

if nargin<2
    q=.05;
end

if nargin<3
    method='pdep';
end

if nargin<4
    report='no';
end

s=size(pvals);
if (length(s)>2) || s(1)>1
    [p_sorted, sort_ids]=sort(reshape(pvals,1,prod(s)));
else
    %p-values are already a row vector
    [p_sorted, sort_ids]=sort(pvals);
end
[dummy, unsort_ids]=sort(sort_ids); %indexes to return p_sorted to pvals order
m=length(p_sorted); %number of tests

if strcmpi(method,'pdep')
    %BH procedure for independence or positive dependence
    thresh=(1:m)*q/m;
    wtd_p=m*p_sorted./(1:m);
    
elseif strcmpi(method,'dep')
    %BH procedure for any dependency structure
    denom=m*sum(1./(1:m));
    thresh=(1:m)*q/denom;
    wtd_p=denom*p_sorted./[1:m];
    %Note, it can produce adjusted p-values greater than 1!
    %compute adjusted p-values
else
    error('Argument ''method'' needs to be ''pdep'' or ''dep''.');
end

if nargout>3
    %compute adjusted p-values; This can be a bit computationally intensive
    adj_p=zeros(1,m)*NaN;
    [wtd_p_sorted, wtd_p_sindex] = sort( wtd_p );
    nextfill = 1;
    for k = 1 : m
        if wtd_p_sindex(k)>=nextfill
            adj_p(nextfill:wtd_p_sindex(k)) = wtd_p_sorted(k);
            nextfill = wtd_p_sindex(k)+1;
            if nextfill>m
                break;
            end
        end
    end
    adj_p=reshape(adj_p(unsort_ids),s);
end

rej=p_sorted<=thresh;
max_id=find(rej,1,'last'); %find greatest significant pvalue
if isempty(max_id)
    crit_p=0;
    h=pvals*0;
    adj_ci_cvrg=NaN;
else
    crit_p=p_sorted(max_id);
    h=pvals<=crit_p;
    adj_ci_cvrg=1-thresh(max_id);
end

if strcmpi(report,'yes')
    n_sig=sum(p_sorted<=crit_p);
    if n_sig==1
        fprintf('Out of %d tests, %d is significant using a false discovery rate of %26.16f\n',m,n_sig,q);
    else
        fprintf('Out of %d tests, %d are significant using a false discovery rate of %26.16f\n',m,n_sig,q);
    end
    if strcmpi(method,'pdep')
        fprintf('FDR/FCR procedure used is guaranteed valid for independent or positively dependent tests\n');
    else
        fprintf('FDR/FCR procedure used is guaranteed valid for independent or dependent tests\n');
    end
end
end














