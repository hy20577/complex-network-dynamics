 
function [normalised_MIR, max_rate] = double_normalised_MIR(Dat, A, opt)

%%%  normalised_MIR --> Matrix whose entries MIR values corresponding pair.
%%% max_rate  --> Optional result (A is necessary) gives the max successful
%%%% rate based on the original network structure.

% Dat is time-series data, including M nodes with n iterations.

% OPTIONAL ARGUMENTS

% A is the adjacency matrix for original network. If A is undefined, not
% define further optional arguments.

% figure1:logical argument(1 or 0). Bar graph and band which shows 100% successful inference if
% exist.

% figure 2:logical argument(1 or 0). Succesful Construction rate based on the paper Successful Inference 

% label --> str; give name of the graphs.
% coupling_strength = Label for coupling strength

% GRAPH showing the relation between GridSize and MIR. 
% opt = [i,j] which is vector with 2 components. here, i and j name of nodes,
% they should be less than or equal to M. (i ~= j) 

time_start = tic;
M = length(Dat(1,:));

% Scaling the Data to interval [0,1]

for i=1:M
    Dat(:,i) = (Dat(:,i) - min(Dat(:,i)))/ (max(Dat(:,i))-min(Dat(:,i)));
end

sum1 = zeros(M,M);
[N_min, N_max] = Grid(Dat);  % Grid sizes satisfy condition

for N=N_min:N_max   %par
    I_s = zeros(M,M);  %% Mutual Information between pairs
    I_c = zeros(M,M);  %% MIR between pairs gaining by dividing MI by t.

    for i=1:M-1  %par
        node1 = Dat(:,i);
        for j=(i+1):M
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
       
             I_s(i,j) = MI;
             I_c(i,j) = MI/t;

        end
    end
    
    MIR_hat = zeros(M,M);
    maks = max(max(I_c));
    minim = min(min(I_c(I_c ~= 0)));

    for l=1:M-1
        for z= l+1:M
            if maks ~= minim
                 MIR_hat(l,z) = (I_c(l,z) - min(min(I_c(I_c ~= 0 )))) ./ (maks - minim);
            end
        end
    end
    
    sum1 = sum1 + MIR_hat;
    
    
    if exist('opt','var')
       if opt(1) > M || opt(2) > M
        warning('Node number cannot be bigger than total number of nodes.')      
       else
       plot(N, I_c(opt(1),opt(2)),'r*')
       xlabel('Grid Size N')
       ylabel('MIR between node1 and node2')
       title('MIR across different Grid Size')
       hold on 
       end
    end
end
fprintf('Elapsed time (function: normalised_MIR) = %s (in DD:HH:MM:SS.MS)\n',datestr(toc(time_start)/(24*60*60),'DD:HH:MM:SS.FFF'))


normalised_MIR = sum1 / max(max(sum1)); 
normalised_MIR = normalised_MIR + normalised_MIR';

% Bar Graph of the MIR_bars for all pairs.

sum1 = 0;
MIR_vals = zeros(M*(M-1)/2,1); 
for i=1:M-1
    for j=i+1:M
        sum1 = sum1+ 1; 
        MIR_vals(sum1) = normalised_MIR(i,j);     
    end
end

if exist('A','var')
    n=100;
    %Th = zeros(n,1);
    recons_perc = zeros(n,1);
     
     for k=1:n
        threshold = k*(1/n);
        Adj_mat = normalised_MIR;
        
        for i=1:M-1
             for j =i+1:M
            
                 if Adj_mat(i,j) < threshold
                     Adj_mat(i,j) = 0;
                     Adj_mat(j,i) = 0; 
                 else
                     Adj_mat(i,j) = 1;
                     Adj_mat(j,i) = 1;
                 end
            
             end
        end

        diff = Adj_mat - A ;
        recons_perc(k) = 100 - (sum(sum(diff == 1 | diff == -1))*(100/sum(sum(A)))); 
%     if sum(sum(Adj_mat == A)) == M*M
%     Th(k) = 1;
%     end   
    end
 
 recons_perc(recons_perc<0) = 0;
 max_rate = max(recons_perc);
end

%     if exist('figure1','var') && figure1~=0
%          h(1) = figure;
%          bar(MIR_vals)
%          title(sprintf(' MIR between pairs for coupling strength = %.2f ', coupling_strength), 'Interpreter','latex', 'FontSize', 20)
% %%%          annotation('textbox',[.9 .5 .1 .2], ...
% %%%        'String',['Length of time series= ' num2str(length(Dat(:,1)))],'EdgeColor','none')
% 
%          if max_rate == 100
%          hold on
%          low_bound = (1/n)*find(recons_perc ==100, 1);
%          up_bound = (1/n)*find(recons_perc ==100, 1, 'last');
%          yline(low_bound,'r-')
%          yline(up_bound, 'r-')
%          region_x = [0, M*(M-1)/2, M*(M-1)/2, 0 ]; 
%          region_y= [low_bound, low_bound, up_bound, up_bound];
%          patch(region_x,region_y,'r','FaceAlpha',0.2)
%          else
%          disp('No interval satisfies 100% succesfully inference')
%          end
%          
%          if exist('label', 'var')
%             saveas(gca, sprintf('%s/MIR_bar for g_l= %.2f_%s.pdf', path, coupling_strength, strcat(label)))
%             else
%             saveas(gca, sprintf('%s/MIR_bar for g_l= %.2f.pdf',path, coupling_strength))
%          end
%     end
%       
%     if exist('figure2','var') && figure2~=0
%        h(2) = figure;
%        plot(linspace(0,1,100),recons_perc)
%        title(sprintf(' MIR between pairs for coupling strength = %.2f ', coupling_strength), 'Interpreter','latex', 'FontSize', 20)
%        xlabel('Threshold','Interpreter','latex', 'FontSize', 20)
%        ylabel('%','Interpreter','latex', 'FontSize', 20)
% %        annotation('textbox',[.9 .5 .1 .2], ...
% %        'String',['Length of time series= ' num2str(length(Dat(:,1)))],'EdgeColor','none')
%        
%         if exist('label', 'var')
%          saveas(gca, sprintf('%s/Reconstruction for g_l= %.2f_%s.pdf',path, coupling_strength, strcat(label)))
%         else
%          saveas(gca, sprintf('%s/Reconstruction for g_l= %.2f.pdf',path, coupling_strength))
%         end
%     end
    
%     writematrix(normalised_MIR, sprintf('%s/normalised_MIR_g_l= %.2f.txt',path, coupling_strength))
%     writematrix(max_rate, sprintf('%s/max_rate_g_l= %.2f.txt',path, coupling_strength))

    
%         if exist('figure1','var') && figure1==1 && exist('figure2','var') && figure2==1
%           savefig(h,['Figures' num2str(g_l) '.fig']); 
%           close(h);
%         end

end


