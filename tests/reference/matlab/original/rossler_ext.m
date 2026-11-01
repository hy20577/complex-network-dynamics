% Extracted unchanged from the local functions of rossler_generator.m so it can be tested on its own.
function output= rossler_ext(t, X, Adj,w, a, f, c, K)

M = size(Adj,2);
x= X(1:M); y = X(M+1:2*M); z=X(2*M+1:3*M);
output = zeros(3*M+9*M^2,1);

 for i=1:M
    sum1 = 0;
        for j=1:M
            sum1 = sum1 + Adj(i,j)*(x(j)-x(i));
        end
    output(i) =  -w(i)*y(i)-z(i)+ K*sum1;
    output(M+i) = w(i)*x(i) + a*y(i);
    output(2*M+i) = f + z(i) * (x(i)-c);
 end

 %%%% Partial derivatives
x_dx = zeros(M,M); x_dy = zeros(M,M); x_dz = zeros(M,M);
y_dx = zeros(M,M); y_dy = zeros(M,M); y_dz = zeros(M,M);
z_dx = zeros(M,M); z_dy = zeros(M,M); z_dz = zeros(M,M);

for i=1:M
    for j=1:M
        if i ==j
        x_dx(i,j) = -K*sum(Adj(i,:));
        else
            x_dx(i,j) = K*Adj(i,j);
        end
    end
end
 for i=1:M
     x_dy(i,i)=-w(i);
 end
for i=1:M
     x_dz(i,i)=-1;
end
for i=1:M
     y_dx(i,i) = w(i);
end
for i=1:M
     y_dy(i,i)=a;
end
for i=1:M
     z_dx(i,i)= z(i);
end
for i=1:M
     z_dz(i,i)= x(i)-c;
end

Jacob_Lorenz = [x_dx, x_dy, x_dz; y_dx, y_dy, y_dz; z_dx, z_dy, z_dz];

Y = zeros(3*M,3*M);
for i=1:3*M
Y(:,i) = X(i*3*M+1:(i+1)*3*M);
end

output(3*M+1:3*M+(3*M)^2)= Jacob_Lorenz*Y;

end











