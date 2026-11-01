function sync = global_sync(Dat)

% Global synchronisation from Kuramoto model, data consists of phases for
% n variables(nodes) with the length T.

    T = size(Dat,1);
    nVar = size(Dat,2);
    sync = zeros(T,1);
    rho = zeros(T,1);
    
        for i=1:T
            sum2 =0;
                for j=1:nVar
                   sum2 = sum2 + exp(1i*Dat(i,j)); 
                end
        sum2 = abs(sum2/(exp(1i*(sum(Dat(i,:))/nVar))));
        rho(i) = (1/nVar)*sum2;
        sync(i) = sum(rho(1:i))/i;
        end         
end

