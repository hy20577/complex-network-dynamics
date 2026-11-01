function pairwise_sync = pairwise_sync_of_phases(Dat)

% From Modular to centralized organisation of synchronization in functional
% areas of cat cerebral cortex, Jesus Gomez-Gardenes et. al., 2010.

M = size(Dat,2);  % number of oscillators
pairwise_sync = zeros(M,M);

        for i =1:M
            for j=i:M
                RP = Dat(:,i) - Dat(:,j);
                pairwise_sync(i,j) = abs(sum(exp(1i*RP(:))))/(length(RP));
                pairwise_sync(j,i) = pairwise_sync(i,j);
            end
        end

end


