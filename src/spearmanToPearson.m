function out = spearmanToPearson(mat)
% Under an assumption of Gaussian data, we can convert 
% a sample spearman rank correlation into a sample 
% Pearson rank correlation using a standard formula
% See https://blogs.sas.com/content/iml/2023/04/05/...
% interpret-spearman-kendall-corr.html
    out = 2*sin(pi*mat/6);
end