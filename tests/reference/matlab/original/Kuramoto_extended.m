% Extracted unchanged from the local functions of Kuramoto_generator.m so it can be tested on its own.
function theta = Kuramoto_extended(t, y, K, A, w)

% A adjacency Matrix
% K is the coupling strength of the model
% y is extended inputs. First M is the initial conditions for M
% oscillators, next ones deviation vectors. 

M = size(A,2); % number of oscillators
theta = zeros(M+M^2,1);
Jacob_mat = zeros(M,M); 
% Computation of orbits and Jacobian

    for i=1:M
        sum1 = 0 ;
        sum2 = 0;
        for j=1:M
            sum1 = sum1+ A(i,j)*sin(y(j)-y(i));
            sum2 = sum2 + A(i,j)*cos(y(j)-y(i));
            if i ~= j
                Jacob_mat(i,j) = (K/M)*A(i,j)*cos(y(j)-y(i));
            end

        end
    theta(i) = w(i) + (K/M)*sum1;  
    Jacob_mat(i,i) = -(K/M)*sum2;
    end


Y = zeros(M,M);
for i=1:M
Y(:,i) = y(i*M+1:(i+1)*M);
end

theta(M+1:M+M^2)= Jacob_mat*Y;


end