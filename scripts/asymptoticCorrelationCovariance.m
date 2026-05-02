rho = [[1 .2 .6 .4];
    [.2 1 .5 .1];
    [.6 .5 1 .2];
    [.4 .1 .2 1]];

rho = randomCorrelationMatrix(10);
sample_size = 10;
k = size(rho,1);

rvec = vecL(rho);

i1=1;
i2=2;
j1=1;
j2=2;
n = sample_size-1;

pf = zeros(k,k,k,k);

for i1=1:k
    for i2=1:k
        for j1=1:k
            for j2=1:k
                pf(i1,i2,j1,j2) = (1/(2*n))*(2*rho(i1,j1)*rho(i2,j2)+...
                                2*rho(i1,j2)*rho(i2,j1)+...
                                -2*rho(j1,j2)*(rho(i1,j1)*rho(i2,j1) ...
                                            +rho(i1,j2)*rho(i2,j2))...
                                -2*rho(i1,i2)*(rho(i1,j1)*rho(i1,j2)...
                                            +rho(i2,j1)*rho(i2,j2))...
                                +rho(i1,i2)*rho(j1,j2)*(rho(i1,j1)^2 ...
                                                       +rho(i1,j2)^2 ...
                                                       +rho(i2,j1)^2 ...
                                                       +rho(i2,j2)^2));
            end
        end
    end
end
% 
% diagonal_parts = [];
% for i=1:k
%     for j=1:k
%     if i~=j
%         diagonal_parts =  [diagonal_parts ;pf(i,j,i,j)];
%     end
%     end
% end
% 
% disp(mean(diagonal_parts))
% disp(std(diagonal_parts))
% 
% 
% off_diagonal_parts = [];
% for i1=1:k
%     for j1=1:k
%         for i2=1:k
%             for j2=1:k
%                 if i1~=j1 && i2~=j2 && ~(i1==j1 && i2==j2)
%                     off_diagonal_parts =  [off_diagonal_parts ;pf(i1,j1,i2,j2)];
%                 end
%             end
%         end
%     end
% end
% 
% disp(mean(off_diagonal_parts))
% disp(std(off_diagonal_parts))


idx = [];
t = 1;
for i=2:k
    for j=1:(i-1)
        idx(t,:) = [j,i];
        t = t + 1;
    end
end

covR = zeros(height(idx),height(idx));
for i=1:size(covR,1)
    for j=1:size(covR,1)
        covR(i,j) = pf(idx(i,1),idx(i,2),idx(j,1),idx(j,2));
    end
end

figure,
imagesc(covR);
title({"Correlation Asymptotic Covariance:",...
strcat("sample size: ",num2str(sample_size)),...
strcat("num variables:",num2str(k))});