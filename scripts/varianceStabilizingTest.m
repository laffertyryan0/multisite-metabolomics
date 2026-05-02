addpath('../src')

% First try to see an example of the variance being
% dependent on the mean

N=10;
k=5;
sample_size = 10;
M = 50; % num groups in each of which all are iid
x = zeros([k,k,N,M]);
z = zeros([k,k,N,M]);
for m=1:M
    mu = mvnrnd(zeros(k,1),eye(k));
    l = 1;
    sig = 5*(l*randomCorrelationMatrix(k)+(1-l)*eye(k));
    for n=1:N
        data = mvnrnd(mu,sig,sample_size);
        x(:,:,n,m) = corr(data,'Type','Pearson');
        z(:,:,n,m) = atanh(x(:,:,n,m));
    end
end


rho = mean(x,3); % mean within each group
v = std(x,[],3);

scatter(reshape(mean(v(:,:,1,:),[1,2]),[M,1]),...
    reshape(mean(rho(:,:,1,:),[1,2]),[M,1]));
xlabel("Mean (Average over entries)")
ylabel("Std (Average over entries)")
title("Correlations of random data")

rho = mean(z,3); % mean within each group
v = std(z,[],3);

% Remove infs and nans
rho(isinf(rho)) = 0;
rho(isnan(rho)) = 0;
v(isinf(v)) = 0;
v(isnan(v)) = 0;

figure,
scatter(reshape(mean(v(:,:,1,:),[1,2]),[M,1]),...
    reshape(mean(rho(:,:,1,:),[1,2]),[M,1]));
xlabel("Mean (Average over entries)")
ylabel("Std (Average over entries)")
title("Stabilized Correlations of random data")
