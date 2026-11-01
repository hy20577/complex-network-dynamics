
function f=lorenz_ext(t,X, Adj, K, sigma, rho, beta)
% K = coupling strength
M = size(Adj,2);
% Parameters
% sigma = 16;  %10;
% rho = 45.92;
% beta = 4; %8/3;

x=X(1:M); y=X(M+1:2*M); z=X(2*M+1:3*M);

f=zeros(3*M+9*M^2,1);

        for i=1:M
            sum1 = 0;
                for j=1:M
                    sum1 = sum1 + Adj(i,j)*(x(j)-x(i));
                end
            f(i) = sigma*(y(i) - x(i)) + K*sum1;
            f(M+i) = x(i)*(rho-z(i))-y(i);
            f(2*M+i) = x(i)*y(i)-beta*z(i);
        end

x_dx = zeros(M,M);
    for i=1:M
        for j=1:M
            if i ==j
            x_dx(i,j) = -sigma-K*sum(Adj(i,:));
            else
                x_dx(i,j) = K*Adj(i,j);
            end
        end
    end

 x_dy = zeros(M,M);
     for i=1:M
         x_dy(i,i)=sigma;
     end
 x_dz = zeros(M,M);
 y_dx = zeros(M,M);
     for i=1:M
             y_dx(i,i) = rho-z(i);
     end
y_dy = zeros(M,M);
    for i=1:M
    y_dy(i,i)= -1;
    end
    y_dz = zeros(M,M);
    for i=1:M
    y_dz(i,i)= -x(i);
    end
z_dx = zeros(M,M);

    for i=1:M
    z_dx(i,i)= y(i);
    end

z_dy = zeros(M,M);

    for i=1:M
    z_dy(i,i)= x(i);
    end
z_dz = zeros(M,M);

    for i=1:M
    z_dz(i,i)= -beta;
    end

    Jacob_Lorenz = [x_dx, x_dy, x_dz; y_dx, y_dy, y_dz; z_dx, z_dy, z_dz];

Y = zeros(3*M,3*M);
for i=1:3*M
Y(:,i) = X(i*3*M+1:(i+1)*3*M);
end

f(3*M+1:3*M+(3*M)^2)= Jacob_Lorenz*Y;

end

