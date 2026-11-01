% Extracted unchanged from the local functions of Mir_surrogate_data.m so it can be tested on its own.
function [TPR,FPR] = TPR_FPR(Orig_Network,Inferred_Network)
TP_SSM = 0;
TN_SSM = 0;
FP_SSM = 0;
FN_SSM = 0;
adj_SSM = Inferred_Network;
A = Orig_Network;
M = size(A,2);

for i=1:M
  for j=1:M

          if adj_SSM(i,j) ==1  && A(i,j)== 1
            TP_SSM = TP_SSM + 1;
          end
           if adj_SSM(i,j) == 1 && A(i,j) == 0  
              FP_SSM = FP_SSM + 1;
          end
          if  adj_SSM(i,j) == 0 && A(i,j) == 0  
              TN_SSM = TN_SSM + 1;
          end
          if  adj_SSM(i,j) == 0 && A(i,j) == 1  
              FN_SSM = FN_SSM + 1;
          end

   end
end

            TPR = TP_SSM/(TP_SSM+FN_SSM);
            FPR = FP_SSM / (FP_SSM + TN_SSM);

end

