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
            sqdiff(j1,j2) = sum((actual{j1}-estimated{j2})^2,'all');
        end
    end

    % Likely order of estimated matrices, found by taking the smallest of 
    % each row in sqdiff, that is, the actual{j} that is closest to the
    % given estimated

    [~,order] = min(sqdiff,[],2);

    for j=1:r
        average_errors(j) = mean(abs(vecL(estimated{j}) - ...
                                vecL(actual{order(j)})),'all');
    end

    if length(unique(order))==r
        matrix_display_order = order;
    else
        matrix_display_order = 1:r;
    end

    % Estimated Spearman
    disp("Estimate: ")
    for j=1:r
        disp(estimated{matrix_display_order(j)}...
            (1:display_width,1:display_width))
    end

    % Actual Pearson
    disp("Actual:")
    for j=1:r
        disp(actual{j}(1:display_width,1:display_width))
    end


    % TODO: There is potentially a problem if order has repeated elements
    % find a solution for this. For now just warn

    if length(unique(order))~=r
        warning("Unable to find order of estimates. Absolute error may be inaccurate. ")
    end

    disp("Average Absolute Errors:")
    disp(average_errors)
    
    disp("=========================================================")

end