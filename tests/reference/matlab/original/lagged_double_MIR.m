
function [lagged_MI, lagged_MIR] = lagged_double_MIR(Dat,max_lag)

time_start = tic;
n = length(Dat(:,1));
M = length(Dat(1,:));
sum1 = zeros(M,M);
sum2 = zeros(M,M);
[N_min, N_max] = Grid(Dat);

parfor N=N_min:N_max   %par
I_s = zeros(M,M);  %% Mutual Information between pairs
I_c = zeros(M,M);  %% MIR between pairs gaining by dividing MI by t.

    for i=1:M-1  %par
        for j=(i+1):M
            
            MI_vals = zeros(2*max_lag+1,1);
            lags = zeros(2*max_lag+1,1);
            
            
           %%%% NEGATIF LAG
           counter = 0;
             
           for tau = 1:max_lag   %% negatif lags (similar with 'xcorr' function) till -(max_lag). Like in xcorr fnc., I may need to consider other side as well.  
                
                counter = counter + 1;
                node1 = Dat(1:(n-tau),i) ; 
                node2 = Dat(tau+1:end,j) ;
                
                lags(counter) =  -1*tau; 
                
                node12 = [node1, node2];
                [coordinates, ppb] = location(node12,N);        
               % t = corr_decay(node12, N); %% correlation decay time for each pair.
               % t = t(t~=0);
       
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

              MI_vals(counter) = MI;  
            end
            
            %%% ZERO LAG
            lags(max_lag+1)= 0;
            
            node1 = Dat(:,i) ; 
            node2 = Dat(:,j) ;
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
               
             MI_vals(max_lag+1) = MI; 
            
            %%% POZITIF LAG
             for tau = 1:max_lag   %%  pozitif lags (Like in xcorr fnc.)  
                
                node1 = Dat(tau+1:end,i) ; 
                node2 = Dat(1:(n-tau),j) ;
                lags(max_lag+1+tau) = tau; 
                
                node12 = [node1, node2];
                [coordinates, ppb] = location(node12,N);        
%                 t = corr_decay(node12, N); %% correlation decay time for each pair.
%                 t = t(t~=0);
%        
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


               id2 = max_lag+1+tau;
                MI_vals(id2) = MI;  
             end

             I_s(i,j) = max(MI_vals);  % taking the maximum of the MI across lags!
             I_c(i,j) = I_s(i,j)/t;  % MIR dividing MI by correlation decay time.
            
             
        end
    end
    
    MIR_hat = zeros(M,M);
    
    for l=1:M-1
        for z= l+1:M
    MIR_hat(l,z) = (I_c(l,z) - min(min(I_c(I_c ~= 0 )))) ./ (max(max(I_c)) - min(min(I_c(I_c ~= 0))));
        end
    end
    
    sum1 = sum1 + MIR_hat;
    sum2 = sum2 + I_s;
end


normalised_MIR = sum1 / max(max(sum1)); 
lagged_MIR = normalised_MIR + normalised_MIR';

lagged_MI = sum2 + sum2';
    
fprintf('Elapsed time (function: lagged_MIR) = %s (in DD:HH:MM:SS.MS)\n',datestr(toc(time_start)/(24*60*60),'DD:HH:MM:SS.FFF'));

    
%         if exist('figure1','var') && figure1==1 && exist('figure2','var') && figure2==1
%           savefig(h,['Figures' num2str(g_l) '.fig']); 
%           close(h);
%         end

end

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
%             % view(biograph(DG,[],'ShowArrows','off'));
%             ShortestPath = graphallshortestpaths(DG);
%             ShortestPath(ShortestPath==Inf) = 0;
%             corr_decay = max(max(ShortestPath));
%         end
%    end
%             
% end

%  function  clf = coupled_logistic_fnc (x,r,A,alpha)  %clf produce 1-step further
% % Input arguments: % x initial values,
% %r constant parameter,
% % A is the adjacency matrix, 16x16 in article.
% % alpha coupled strength
% global r A alpha
% 
% M= length(A(:,1));   %% Number of nodes, in other words, length of A.
% y = zeros(M,1);    
% 
%     for i=1:M %rows loop
%         sum1=0;
%         for j=1:M %columns loop
%             sum1 = sum1 + A(i,j) * logis(r, x(j));
%         end
%         coupling_term = sum1 * (alpha/sum(A(i,:)));
%         y(i) = logis(r, x(i)) * (1-alpha) + coupling_term;
%     end
%     
%     function logis = logis(r, x)
%        logis = r*x*(1-x); 
%     end
% clf = y;
% end
%  
% function Jacob_log= Jacob_log(x,r ,A, alpha)
% %Jacob_log calculates the jacobian matrix in the given set of points
% % x is vector of initial points.
% global r A alpha
% k = length(x);
% Jacob_log = zeros(k,k);
%     for i=1:k
%         for j=1:k
%             if i == j
%                 Jacob_log(i,j) = df_log(r,x(i))*(1-alpha)+ (alpha/sum(A(i,:)))*A(i,j)*df_log(r,x(j));
%             else
%                 Jacob_log(i,j) = (alpha/sum(A(i,:)))*A(i,j)*df_log(r,x(j));
%             end
%         end
%     end
%     
%     function df_log = df_log(r,x)
%     df_log = r*(1-2*x);
%     end
% end
