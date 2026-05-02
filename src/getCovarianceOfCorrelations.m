function covR = getCovarianceOfCorrelations(rho,sample_size)
    k = size(rho,1); % matrix width
    n = sample_size - 1; % n used in P&F expression
    covR = zeros(k*(k+1)/2,k*(k+1)/2);
    % covR = cov(vecL(r),vecL(r))

    
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

    

end