function [Orbits, Lya_overtime] = Composite_log_generator(Adjacency, alpha, x0, r, n, niter, time_window, std_threshold, max_discard, savefigs)

% n is the order of composite logistic function !!!

if max_discard < time_window
    max_discard = time_window;
    warning('max_discard = time_window set up automatically')
end


k = length(x0);
w = eye(k);
v = x0;
sum1 = 0;
Orbits = zeros(niter, k);

    for i = 1:time_window
        K = zeros(k,k);   % matrix stores deviation vectors in columns.
            J = Jacob_log(v, r, n, Adjacency, alpha); 

            for j =1:k
            K(:,j) = J*w(:,j);  % images of w() under Jacobian matrix.  
            end
            
            ort_K = gramschmidt(K);  %% gs function produces orthonormal vectors.
            y = zeros(1,k);
            
            for j=1:k
            y(j) = norm(ort_K(:,j));
                if sum(ort_K(:,j)) ~= 0 
                    w(:,j) = ort_K(:,j)/norm(ort_K(:,j));
                else
                    w(:,j) = 0;
                end
            end
            sum1 = sum1 + log(y);
            Lya_overtime(i,:) = sort(sum1/i, 'descend');
            v= coupled_logistic_fnc(v,r,Adjacency,alpha);
    end

     sum2 = 0;

    while std(Lya_overtime(end-time_window+1:end,1)) > std_threshold 
        sum2 = sum2+1;
        K = zeros(k,k);   % matrix stores deviation vectors in columns.
        J = Jacob_log(v, r, n, Adjacency, alpha); 
                for j =1:k
                K(:,j) = J*w(:,j);  % images of w() under Jacobian matrix.  
                end
            ort_K = gramschmidt(K);  %% gs function produces orthonormal vectors.
            y = zeros(1,k);

            for j=1:k
            y(j) = norm(ort_K(:,j));
                if sum(ort_K(:,j)) ~= 0 
                    w(:,j) = ort_K(:,j)/norm(ort_K(:,j));
                else
                    w(:,j) = 0;
                end
            end

            sum1 = sum1 + log(y);
            Lya_overtime(time_window+sum2,:) = sort(sum1/(sum2+time_window), 'descend');
            v= coupled_logistic_fnc(v,r,Adjacency,alpha);

            if sum2-time_window > max_discard 
                warning(sprintf('std_threshold is too small to stabilize; first %d discarded', time_window+max_discard))
                break
            end

    end
    
    for i = 1:niter
   
            Orbits(i,:) = v;
            K = zeros(k,k);   % matrix stores deviation vectors in columns.
            J = Jacob_log(v, r, n, Adjacency, alpha); 

            for j =1:k
            K(:,j) = J*w(:,j);  % images of w() under Jacobian matrix.  
            end
            
            ort_K = gramschmidt(K);  %% gs function produces orthonormal vectors.
            y = zeros(1,k);
            
            for j=1:k
            y(j) = norm(ort_K(:,j));
                if sum(ort_K(:,j)) ~= 0 
                    w(:,j) = ort_K(:,j)/norm(ort_K(:,j));
                else
                    w(:,j) = 0;
                end
            end
  
            sum1 = sum1 + log(y);
           
            Lya_overtime(i+time_window+sum2,:) = sort(sum1/(i+time_window+sum2), 'descend'); 
            
            v= coupled_logistic_fnc(v,r,Adjacency,alpha);
            
        if  mod(i,1e3) == 0
            fprintf('Completed job : %s %% \n', floor(num2str(i*100/niter)))
        end
        
        
    end


    if savefigs == 1 
        currpath = pwd;
        if ~exist(sprintf('%s/Logistic_Info', currpath), 'dir')
            mkdir(sprintf('%s/Logistic_Info', currpath))
        end
        
        info = {'discarded', time_window+sum2; 'MLE', Lya_overtime(end,1); 'niter', niter;'std_threshold', std_threshold; ...
          'coupling strengtht', alpha; 'time window', time_window; 'max_discard', max_discard};
        writecell(info, sprintf('Logistic_Info/info_%.2f.txt', alpha))
        
        writematrix(Lya_overtime(:,1), sprintf('Logistic_Info/MLE_%.2f.txt', alpha))
        
        discard = length(Lya_overtime(:,1)) - niter;
        figure;
        loglog(Lya_overtime(:,1))
        xlim([0 length(Lya_overtime(:,1))])
        ylim([-abs(min(Lya_overtime(:,1))), 1.5*max(Lya_overtime(:,1))])
        xline(discard, 'r--')
        yline(0, 'r-')
        hold on
        patch([0 discard discard 0], [-abs(min(Lya_overtime(:,1))) -abs(min(Lya_overtime(:,1))) 1.5*max(Lya_overtime(:,1)) 1.5*max(Lya_overtime(:,1))],'red', 'FaceAlpha', 0.3)
        t = text((discard)/2,-abs(min(Lya_overtime(:,1)))/2,'Transient Period');
        t.Rotation = 90;
        hold off
        saveas(gca, sprintf('Logistic_Info/MLE_over_Time_%.2f.png', alpha))
        close all
    end
end

%% %%%%%%%%%%%%%%%%%%%%%%%  FUNCTIONS  %%%%%%%%%%%%%%%%%%%%%


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
            if a ~= 0 
                c = c - (a*a'*b)/(a'*a);
            end
        end
        U(:,ii) = c;
    end
end

function Jacob_log= Jacob_log(x,r, n, A, alpha)
%Jacob_log calculates the jacobian matrix in the given set of points
% x is vector of initial points.
k = length(x);
Jacob_log = zeros(k,k);
    for i=1:k
        for j=1:k
 
                if i == j
                    Jacob_log(i,j) = df_log(r,x(i),n)*(1-alpha);
                else
                    if sum(A(i,:)) ~= 0
                    Jacob_log(i,j) = (alpha/sum(A(i,:)))*A(i,j)*df_log(r,x(j), n);
                    else 
                    Jacob_log(i,j)= 0;
                    end
                end
        end
    end
    
    function df_log = df_log(r,x, n)
    df_log = 1;
       for deg = 1:n-1
          df_log = df_log * df(r,comp_logis(r, x, deg));
       end
        df_log = df(r, x)*df_log;
    end

    function df = df(r, x)
        df = r - 2*r*x;
    end
end

function  clf = coupled_logistic_fnc (x,r,A,alpha)  %clf produce 1-step further
% Input arguments: % x initial values,
%r constant parameter,
% A is the adjacency matrix, 16x16 in article.
% alpha coupled strength

M= length(A(:,1));   %% Number of nodes, in other words, length of A.
clf = zeros(M,1);

    for i=1:M %rows loop
        sum1=0;
        for j=1:M %columns loop
            sum1 = sum1 + A(i,j) * comp_logis(r, x(j), 3);
        end
        if sum(A(i,:)) ~= 0
            coupling_term = sum1 * (alpha/sum(A(i,:)));
        else
             coupling_term = 0;   
        end
        clf(i) = comp_logis(r, x(i), 3) * (1-alpha) + coupling_term;
    end

end

function logis = comp_logis(r, x, n) % n is the number of order of composite function !
    for i=1:n
        x = r*x*(1-x); 
    end
    logis = x;
end


