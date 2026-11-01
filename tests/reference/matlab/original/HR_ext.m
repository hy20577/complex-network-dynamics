
function f=HR_ext(t,y, A, g_l)
% We need the extended function to evolve deviation vectors in odesolver
% simultaneously. Detail in Chaos chapter 9. 

%%% Get ready inputs
M=size(A,1);
D = zeros(M,M);   %%% diagonal matrix of degree nodes in eq.9

    for j=1:M
    D(j,j) = sum(A(j,:));    
    end

C = D - A;   %%% Laplacian Matrix
a=1;
b=3;
d=5;
s=4;
h = 0.005;

f= zeros(12,1);

%%% Ode part
 p_dot = zeros(1,M);
 q_dot = zeros(1,M);
 n_dot = zeros(1,M);
    for i=1:M

        sum1 = 0;
            for l=1:M
            sum1 = sum1 + C(i,l)*y(l);
            end
        
        p_dot(i)  =  y(i+M) - y(i)^3 + 3*y(i)^2 - y(i+2*M) + 3.25 - g_l*sum1; 
        q_dot(i)  =  1 - 5*y(i)^2 - y(i+M); 
        n_dot(i)  =  0.02*(y(i)+1.6) - 0.005*y(i+2*M);
    end
    f(1:3*M) = [p_dot q_dot  n_dot];
    
% Jacobian Matrix

p_dp = zeros(M, M);
for j=1:M
    for i=1:M
        if i == j
         p_dp(i,j) = -3*a*y(i)^2 + b*y(i) - g_l* C(i,i);
        else
         p_dp(i,j) = - g_l*C(i,j);
        end
    end
end

q_dp = zeros(M,M);
    for i=1:M
        q_dp(i,i) = -2*d*y(i);
    end

n_dp = zeros(M,M);
    for i= 1:M
        n_dp(i,i)= h*s;
    end
p_dq =eye(M);
q_dq = -eye(M);
n_dq = zeros(M,M);
p_dn = -eye(M);
q_dn = zeros(M,M);
n_dn = -h*eye(M);

Jacob_HR = [p_dp p_dq p_dn; q_dp q_dq q_dn; n_dp n_dq n_dn]; % 36x36 Jacobian
Y = zeros(3*M,3*M);

for i=1:3*M
Y(:,i) = y(i*3*M+1:(i+1)*3*M);
end

f(3*M+1:3*M+(3*M)^2)= Jacob_HR*Y;
end