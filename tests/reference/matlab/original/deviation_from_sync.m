function  E = deviation_from_sync(x1, x2, tvalues)
% Average deviation from the sync using the values of 2 oscillators,
% time values.

n = length(tvalues)-2;
array_deviation_over_time = zeros(n,1);
sum1 =0;

    for i=1:n
        sum1 = sum1 + abs(x2(i) - x1(i));
        if i>1
            array_deviation_over_time(i-1) = (1/tvalues(i))*sum1;
        end
    end

E = array_deviation_over_time;
end