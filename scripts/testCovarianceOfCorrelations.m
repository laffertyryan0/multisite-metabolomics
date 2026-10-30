addpath("../src")
theta0 = [1 .7 .2 .3 .4; 
          0 1 -.5 .2 -.2;
          0 0   1 .1 .2;
          0 0  0  1  -.3;
          0 0 0 0 1];
theta0 = theta0(1:4,1:4)
theta0 = theta0 + theta0' - eye(length(theta0));
%theta0 = ones(5,5);
%theta0 = eye(3,3);
theta1 = nearcorr(theta0);

d = length(theta0);

rs = [];
r0s = [];

n=100;

for mc = 1:10000
    
    data = mvnrnd(zeros(d,1),theta1,n);
    
    samplecorr = corr(data,'Type','Pearson');
    
    r = vecL(atanh(samplecorr));
    r0 = vecL(samplecorr);
    rs(mc,:) = r;
    r0s(mc,:) = r0;
end

cov1 = getCovarianceOfCorrelationsFisher(atanh(theta1),n);
cov01 = getCovarianceOfCorrelations(theta1,n);
cov2 = cov(rs);
cov02 = cov(r0s);

disp(cov1)
corrcorr = inv(diag(diag(cov1))^(1/2))*cov1*inv(diag(diag(cov1))^(1/2));

