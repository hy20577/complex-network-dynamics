
function [MI, P_null] = Mutual_info(DAT, grid_size)
% grid_size = [10,8]; % 10-by-8 partition
M = size(DAT,2);
P_null = zeros(M,M);
MI = zeros(M,M);

for N1 = 1:M
    for N2 = (N1+1):M 
            node1 = DAT(:,N1);
            node2 = DAT(:,N2);
            n = length(node1);
            
            if length(grid_size) == 1  % equal number of partition for two timeseries
            Nx =  grid_size;
            Ny =  grid_size;
            else             %% nonequal number of partition; N_x != N_y
            Nx = grid_size(1);
            Ny = grid_size(2);
            end
            
            intervals1 = linspace(min(node1), max(node1), Nx+1);
            intervals2 = linspace(min(node2), max(node2), Ny+1);
            
            % number of points in the partition.
            o_xy = zeros(Nx, Ny);
            E_xy = zeros(Nx, Ny); % Expected value
            
            for i = 1:Nx  % first partition
                for j=1:Ny  % 2nd partition;  
                   o_xy(i,j) = sum(intervals1(i)<= node1 & node1<intervals1(i+1) & ...
                                   intervals2(j)<= node2 & node2<intervals2(j+1));
                end
            end
            
            o_x = sum(o_xy,2);
            o_y = sum(o_xy);
            
            p_xy = o_xy./n;
            p_x = o_x./n;
            p_y = o_y./n;
            
            % Computation of MI
            
            I_xy = 0;
            for i =1:Nx
                for j =1:Ny
                    if p_xy(i,j)~=0
                     I_xy = I_xy + p_xy(i,j)*log(p_xy(i,j)/(p_x(i)*p_y(j)));
                    end
                     E_xy(i,j) = n*p_x(i)*p_y(j); %% expected number of pts in cells.
                end
            end

            MI(N1,N2) = I_xy; 
            
            %% Statistical assessment of Mutual Information
            
            % H_0 : There is no relation between X and Y. 
            
            % Expectation, X and Y independent so, p(x,y) = p(x)*p(y). 
            % E_xy{i,j} = n*p_x{i}*p_y{j}
            
            if (sum(sum(E_xy >= 1)) == Nx*Ny) && (sum(sum((E_xy >= 5))) >= Nx*Ny*0.8)
            warning(sprintf('Cochran Criterion does satisfy for statistical efficiency.(N = %d)', grid_size))
            end
            
            chi_square = 0;
            
            for i=1:Nx
               for j =1:Ny
                    chi_square =  chi_square + (o_xy(i,j) - E_xy(i,j))^2/E_xy(i,j);
               end
            end
            
            v = (Nx-1)*(Ny-1);  % Degree of freedom.
            
            P_null(N1,N2) = gammainc(v/2, chi_square/2); 

    end
end
end



