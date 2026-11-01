function R = globalorder_map(Dat)
% Dat is M-by-n matrices, where M is the number of nodes, n is the
% number of iteration.

% Eq3. in the supplementary material: exact detection of direct links in
% networks of interacting dynamical units
% Nicolas Rubido et. al.

X = Dat;  
[n, M] = size(Dat);
X_sq = X.^2;  
X_avg = mean(X,2);   % average of nodes at time t.

R = zeros(n,1);

    for t = 1:n
        nominator = (sum(X_avg(1:t).^2)/t) - mean(X_avg(1:t))^2;
        dummy= (sum(X_sq(1:t,:))/t)-(sum(X(1:t,:))/t).^2;  % M values
        denominator = sum(dummy)/M;  % average over network.
        R(t) = nominator/denominator;
        
    end
end