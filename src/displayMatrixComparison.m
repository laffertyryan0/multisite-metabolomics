function displayMatrixComparison(rho_est, ...
                                 rho_actual, ...
                                 display_width)
    k = size(rho_est{1,1},1);
    r = length(rho_est);
    display_width = min(display_width,k);


    disp("=========================================================")
    disp("Comparison of Estimated and Actual Correlation Matrices: ")
    fprintf("\nNumber of entries shown: %d x %d \n\n",display_width,display_width)

    % Estimated Spearman
    disp("Estimate: ")
    for j=1:r
        estimated = vecLInverse(rho_est{1,j});
        disp(estimated(1:display_width,1:display_width))
    end

    % Actual Pearson
    disp("Actual:")
    for j=1:r
        actual = rho_actual{1,j};
        disp(actual(1:display_width,1:display_width))
    end
    
    disp("=========================================================")

end