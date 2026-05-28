function out = pearsonToSpearman(mat)
% Under an assumption of Gaussian data, we can convert 
% a sample pearson rank correlation into a sample 
% spearman rank correlation by inverting the standard formula
% See https://blogs.sas.com/content/iml/2023/04/05/...
% interpret-spearman-kendall-corr.html
    out = asin(mat/2)*6/pi;
end
