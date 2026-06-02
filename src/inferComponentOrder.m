function order = inferComponentOrder(estimated, ...
                                     actual)
    % Find correspondence between actual and estimated matrices
    % by minimizing the sum of squared differences for each 
    % index


    r = length(estimated);
    
    sqdiff = zeros(r,r);
    
    for j1=1:r
        for j2=1:r
            sqdiff(j1,j2) = sum((actual{j2}-estimated{j1})^2,'all');
        end
    end

    % Likely order of estimated matrices, found by taking the smallest of 
    % each row in sqdiff, that is, the actual{j} that is closest to the
    % given estimated. If a column has already been taken, take the second
    % smallest, then the third smallest, and so on. There can only be one
    % column taken by each row. 

    order = []; % a column selected for the jth row

    for j_row=1:r
        row = sqdiff(j_row,:);
        row(ismember(1:r,order)) = Inf;
        [~,min_idx_for_this_row] = min(row,[],2);
        order(j_row) = min_idx_for_this_row;
    end
end
