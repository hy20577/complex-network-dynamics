function [tvalues,Orbits, LYA] = Euler(extended_ode, t0, x0, tf, dt, varargin)

%% extended_ode is the function of equation of motion and its Jacobian.
% M = number of oscillator.
% D = dimension of the system; for HR 3, Lorenz 3, Kuramoto 1.

k = ceil((tf-t0)/dt)+1;
tvalues = zeros(k,1);

n = length(x0); % dimension of extended ode function.
sum1= 0;

% Setting initial conditions
t= t0;
y0 = zeros(1, n^2+n);
y0(1:n) = x0;

for i=1:n
    y0((n+1)*i) = 1; % initial deviation vectors: unit vectors.
end
X = y0;

Orbits = zeros(k,n);
LYA = zeros(k,n);
start_time = tic; 
f = waitbar(0,'1');

for i = 1:k
    k1 = extended_ode(t,         X,  varargin{:});
%     k2 = extended_ode(t+dt/2,    X+k1*dt/2,  varargin{:});
%     k3 = extended_ode(t+dt/2,    X+k2*dt/2,  varargin{:});
%     k4 = extended_ode(t+dt,      X+k3*dt,  varargin{:});

    y0 = X +dt*k1;

    Orbits(i,:) = y0(1:n);
    y = y0(1:n);
    tvalues(i) = t;

    K = reshape(y0(n+1:n^2+n),n,n); % variation vectors   
    K = gramschmidt(K);    %% Orthogonalization
        
    norm_K = zeros(1, n);
    NK = zeros(n,n);
        
    for j=1:n   % normalize variation vectors
        norm_K(j) = norm(K(:,j));
        NK(:,j) = K(:,j)/norm_K(j);
    end
       
    sum1 = sum1 + log(norm_K);
    t = t+dt;     
    LYA(i,:) = sort(sum1/(t-t0), "descend"); 
    y(n+1:n^2+n) = reshape(NK,1,n^2);
    X =y;
    
    waitbar(i/k, f, sprintf('approximately %.2f secs remain',(k/i)*toc(start_time)))
end
end

function U = gramschmidt(V)
% GRAM_SCHMIDT - Classic Gram-Schmidt Process
%   Input vectors should be the columns of input matrix.
%   Output = unitary, orthogonal vectors in columns of matrix

n = size(V, 2); % number of columns

U(:,1) = V(:,1);

% find next orthogonal column
for ii = 2:n
    b = V(:,ii);
    c = b;
    for k = 1:ii-1
        a = U(:,k);
        c = c - (a*a'*b)/(a'*a);
    end
    U(:,ii) = c;
end
end