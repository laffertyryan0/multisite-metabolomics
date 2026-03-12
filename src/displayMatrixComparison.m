function displayMatrixComparison(rho_est, ...
                                 rho_actual, ...
                                 display_width)
    
    k = size(rho_est{1,1},1);
    r = length(rho_est);
    display_width = min(display_width,k);


    disp("=========================================================")
    disp("Comparison of Estimated and Actual Correlation Matrices: ")
    fprintf("\nNumber of entries shown: %d x %d \n\n",display_width,display_width)

    estimated = {};
    actual = {};

    average_errors = [];

    % Estimated spearman and actual Pearson 
    for j=1:r
        estimated{j} = vecLInverse(rho_est{1,j});
        actual{j} = rho_actual{1,j};
    end

    
    % Find correspondence between actual and estimated matrices
    % by minimizing the sum of squared differences for each 
    % index

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

    for j=1:r
        average_errors(j) = mean(abs(vecL(estimated{j}) - ...
                                vecL(actual{order(j)})),'all');
    end

    % Estimated Spearman
    disp("Estimate: ")
    for j=1:r
        disp(estimated{order(j)}...
            (1:display_width,1:display_width))
    end

    % Actual Pearson
    disp("Actual:")
    for j=1:r
        disp(actual{j}(1:display_width,1:display_width))
    end

    disp("Average Absolute Errors:")
    disp(average_errors)
    
    disp("=========================================================")

end