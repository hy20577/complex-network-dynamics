

function corr_decay = corr_decay(Dat,gridsize)
% Correlation decay time for all pairs (columns) in given data.
% Inputs 
% Dat: time-series data.  gridsize: number of columns or rows in partition, N.

M=length(Dat(1,:));
   for i=1:M
        for j=i+1:M
            node1 = Dat(:,i); 
            node2 = Dat(:,j);
        
            [coordinates, ~] = location([node1 node2],gridsize);
            
            DG = full(sparse(coordinates(1:end-1,4), coordinates(2:end,4), 1,gridsize^2,gridsize^2)); % Adjacency matrix of sparse graph
            DG(DG ~= 0) =1;        
                for ind = 1:gridsize^2
                    DG(ind,ind) = 0;   % eliminate self connections.
                end
            [n1, n2] = find(DG == 1);
            G = graph(n1,n2,1);
            ShortestPath = distances(G);
            ShortestPath(ShortestPath==Inf) = 0;
            corr_decay = max(max(ShortestPath));
        end
   end
            
end
