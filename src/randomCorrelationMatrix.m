function out = randomCorrelationMatrix(sz)
    % Generate a random correlation matrix for initializing 
    % simulations. This is to make sure the correlation matrix
    % chosen is arbitrary. 
    %
    % sz: height and width of desired matrix
    %
    % out: a random correlation matrix of size sz x sz
    shift = rand*2-1;
    A = shift+rand(sz, sz);
    S = A' * A;
    d = sqrt(diag(S)); 
    out = S ./ (d * d'); 
    out(1:sz+1:end) = 1;