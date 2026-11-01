function [TPR_FPR_ROC, TPR_FPR_ROC_best]  = best_case_TPR_FPR(similarity_matrix,A, plot_inf_net, plot_Roc_plot, barplot, label)

%%% similarity matrix contains of the SSM values between links in bidirectional network.
% Threshold values start with max SSM abn compute TPR, FPR and distance 
%%% A original network.


% similarity matrix M by M.
% A is the original adjacency matrix
% label: which kind of similarity measure uses? It would be appear on the
% plot.



M = size(similarity_matrix,1);
%%%%% obtaining vector containing the similarity values
sim_vals = zeros(M*(M-1)/2,3);
sum1=0;

for diag=1:M
   similarity_matrix(diag,diag)=0; 
end

if  similarity_matrix == triu(similarity_matrix)
    similarity_matrix = similarity_matrix + similarity_matrix';
end


for i=1:M-1
    for j=i+1:M
        sum1 = sum1+1;
        sim_vals(sum1,1) = i;
        sim_vals(sum1,2) = j;
        sim_vals(sum1,3) = similarity_matrix(i,j);     
    end
end
eps = 1e-12;
ordered_ssm = sort(sim_vals(:,3), 'descend') - eps;
nopp = length(ordered_ssm); % nopp --> number of possible pairs.

TPR_SSM = zeros(nopp,1);
FPR_SSM = zeros(nopp,1);
ROC = zeros(nopp,1);

    for k = 1: nopp
        adj_SSM = zeros(M, M);
        
            % creating adj. matrix for inferred network.
            
        for i=1:M
            for j=1:M 
                    if similarity_matrix(i,j) > ordered_ssm(k)
                    adj_SSM(i,j) = 1;
                    else 
                    adj_SSM(i,j) = 0;
                    end 
            end
        end

            TP_SSM = 0;
            TN_SSM = 0;
            FP_SSM = 0;
            FN_SSM = 0;

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

            TPR_SSM(k) = TP_SSM/(TP_SSM+FN_SSM);
            FPR_SSM(k) = FP_SSM / (FP_SSM + TN_SSM);
            ROC(k) = sqrt((TPR_SSM(k)-1)^2+(FPR_SSM(k))^2); % distance to the point (0,1).
     end

        TPR_FPR_ROC = [TPR_SSM,FPR_SSM,ROC, ordered_ssm];
        TPR_FPR_ROC = array2table(TPR_FPR_ROC, 'VariableNames', {'TPR', 'FPR', 'ROC_distance', 'Threshold'}); 
     
        [minimum, ~] = min(TPR_FPR_ROC{:,3}); % best case indices.
        ind = find(TPR_FPR_ROC{:,3} == minimum);
%          up_bound = ordered_ssm(ind(1));
%          low_bound = ordered_ssm(ind(end)+1);
%         
        TPR_FPR_ROC_best = TPR_FPR_ROC{ind, :};
        TPR_FPR_ROC_best = array2table(TPR_FPR_ROC_best, 'VariableNames', {'TPR', 'FPR', 'ROC_distance', 'Threshold'}); 

%          if minimum == 0
%             figure
%             bar(sim_vals)
%             title(sprintf(' %1$s of Pairs', label), 'Interpreter','latex', 'FontSize',20);
%             ylabel('Values')
%             xlabel('Pairs of nodes')
%             hold on
%             yline(low_bound,'r-')
%             yline(up_bound, 'r-')
%             region_x = [0, M*(M-1)/2, M*(M-1)/2, 0 ];
%             region_y= [low_bound, low_bound, up_bound, up_bound];
%             patch(region_x,region_y,'r','FaceAlpha',0.2)
%         else
%             disp('No Perfect Inference')
%         end
        
           th = TPR_FPR_ROC_best.Threshold;
           inf_adj = similarity_matrix;

           for i=1:M-1
                for j=i+1:M
                    if   similarity_matrix(i,j) < th  
                        inf_adj(i,j) = 0;
                        inf_adj(j,i) = 0;
                    else
                        inf_adj(i,j) = 1;
                        inf_adj(j,i) = 1;
                    end
                end
           end
            

             if exist('barplot','var')
                if barplot == 1
                   
                    %%%% barplot of MIRs %%%%%%%%%%
                    counter = 0;
                    connected_nodes = zeros(sum(sum(A))/2,1); 
                    
                    for i=1:M
                        for j=i+1:M
                            if A(i,j) ==1
                                counter = counter+1;
                                connected_nodes(counter,1) = i;
                                connected_nodes(counter,2) = j;
                            end
                        end
                    end


                    %figure
                    sim_vals = sortrows(sim_vals,3);
                    b= bar(sim_vals(:,3));
                    b.FaceColor = 'flat';
                    %b.EdgeColor = "none";
                    b.EdgeAlpha = 0.3;
                    % b.BarWidth = 2;
                    b.FaceAlpha = 0.9;
                    ax = gca;
                    ax.FontSize = 20;
                    

                    ax.XTick = [0, 25, 75, 120]; %[linspace(25,nopp-mod(nopp,50)*2, 3),nopp];
                    
                    k = size(connected_nodes,1);
                    idx_connected = zeros(k,1);

                    for i=1:k
                            idx_connected(i) = find(sum(connected_nodes(i,1:2) == sim_vals(:,1:2), 2) == 2);
                            b.CData(idx_connected(i),:) = [.5 0 .5];
                    end

                    title(sprintf('%s', label), 'Interpreter', 'latex', 'FontSize',12);
%                     ylabel('Similarity Measure','Interpreter','latex', 'FontSize',20)
%                     xlabel('Pairs of nodes','Interpreter','latex', 'FontSize',20)
                    set(gcf,'color','w'); 

                     if minimum == 0 
                        up_bound = ordered_ssm(ind(1));
                        if (ind(end)+1 < length( ordered_ssm)) || (ind(end)+1 == length( ordered_ssm))
                        low_bound = ordered_ssm(ind(end)+1);
                        else
                        low_bound = ordered_ssm(ind(end));
                        end
                        hold on
                        yline(low_bound,'k-')
                        yline(up_bound, 'k-')
                        region_x = [0, M*(M-1)/2, M*(M-1)/2, 0 ];
                        region_y= [low_bound, low_bound, up_bound, up_bound];
                        patch(region_x,region_y,'k','FaceAlpha',0.05)
                     else
                        % yline(th, 'k--')
                        disp('No Perfect Inference')
                     end
                end    
              end
             
            
             if exist('plot_Roc_plot','var')
                if plot_Roc_plot == 1
                    figure
                    plot(FPR_SSM,TPR_SSM,'.')
                    hold on
                    plot(FPR_SSM(ind),TPR_SSM(ind),'r*')
                    grid
                    title(sprintf('FPR vs TPR %s',label), 'Interpreter','latex', 'FontSize',12);
                    ylabel('TPR', 'Interpreter','latex', 'FontSize', 12)
                    xlabel('FPR', 'Interpreter','latex', 'FontSize',12)
                    set(gcf,'color','w'); 
                end
             end
                    %%%%%%%%%%%%%% Inferred network topology  %%%%%%%%%%%%%
                    
            if exist('plot_inf_net','var')
                if plot_inf_net == 1
                    figure
                    g = plot(graph(inf_adj), 'MarkerSize',40, 'Layout','circle');
                    g.EdgeColor = '#0938e0';
                    g.LineWidth = 2; 
                    g.NodeFontSize = 20;
                    g.NodeColor = '#FFA500';
                    g.NodeFontSize = 30; 
                    g.NodeFontAngle = 'normal'; 
                    text(g.XData-0.015, g.YData, g.NodeLabel, 'FontSize',30)
                    g.NodeLabel = {};
                    title('Inferred Network', 'Interpreter','latex', 'FontSize',12);
                    set(gcf,'color','w'); 
                end
            end
                    
                    
                    % 
%                     figure
%                     stairs(ordered_ssm,TPR_FPR_ROC{:,3})
%                     xlabel('Threshold')
%                     title(sprintf('ROC curve of %1$s',label), 'Interpreter','latex', 'FontSize',20);
%                     ylabel('ROC Distance')
%                     hold on
%                     plot(up_bound,TPR_FPR_ROC{ind,3},'r*')
%                     plot(low_bound,TPR_FPR_ROC{ind,3},'r*')
%                  
    
                 
                  
end
