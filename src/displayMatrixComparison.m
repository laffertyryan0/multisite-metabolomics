function [...
          order]...
            =   ...
                    displayMatrixComparison(rho_est, ...
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
    
    order = inferComponentOrder(estimated,actual);
    
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