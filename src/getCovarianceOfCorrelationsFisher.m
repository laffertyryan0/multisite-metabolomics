function covRFisher = getCovarianceOfCorrelationsFisher(rho,sample_size)
    k = size(rho,1); % matrix width
    n = sample_size - 1; % n used in P&F expression
    covRFisher = zeros(k*(k-1)/2,k*(k-1)/2);
    % covR = cov(vecL(r),vecL(r))
    rho_raw = tanh(rho);
    rho_raw = rho_raw - diag(diag(rho_raw)) + eye(length(rho_raw));
    covR = getCovarianceOfCorrelations(rho_raw,sample_size);

        
    idx = [];
    t = 1;
    for i=1:(k-1)
        for j=(i+1):k
            idx(t,:) = [i,j];
            t = t + 1;
        end
    end
    
    covRFisher = zeros(k*(k-1)/2,k*(k-1)/2);
    for i=1:size(covRFisher,1)
        for j=1:size(covRFisher,1)
            covRFisher(i,j) = n*covR(i,j)/...
                                ((sample_size-3)*...
                                    (1-rho_raw(idx(i,1),idx(i,2))^2)...
                                      * (1-rho_raw(idx(j,1),idx(j,2))^2));
        end
    end

    

end