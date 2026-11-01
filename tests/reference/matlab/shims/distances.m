function D = distances(G)
% Minimal stand-in for MATLAB's distances() on an unweighted graph:
% all-pairs shortest path lengths by breadth-first search, Inf if unreachable.
n = G.n;
A = false(n, n);
if n > 0
    A(sub2ind([n n], G.s, G.t)) = true;
end
A = A | A';
D = inf(n, n);
for src = 1:n
    D(src, src) = 0;
    frontier = src;
    d = 0;
    while ~isempty(frontier)
        d = d + 1;
        nxt = find(any(A(frontier, :), 1) & isinf(D(src, :)));
        D(src, nxt) = d;
        frontier = nxt;
    end
end
end
