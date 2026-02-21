function tests = argmaxQTest
addpath("../src")
tests = functiontests(localfunctions);
end

% Make sure it runs and bound the time it takes
function testTimeTaken(testCase)
    r = 5;
    k = 50;
    L = 200;
    n_subjects = ones(L,1)*500;
    
    % Initialize EM parameters
    alpha_est = rand(1,r);
    alpha_est = alpha_est/sum(alpha_est); % Random initialization
    rho_est = cell(1,r);
    sigma_rho_est = cell(1,r);
    for j = 1:r
        rho_est{j} = vecL(randomCorrelationMatrix(k)); % Random initialization
        sigma_rho_est{j} = .1*speye(k*(k-1)/2);
    end
    
    w = mean(sqrt(n_subjects))./sqrt(n_subjects); % Lab-wise weighting factor for variances (L vector)
    
    GD_LEARNING_RATE=.001;
    MAX_GD_ITERATIONS=10;
    GD_TOLERANCE=.001;
    INIT_GDVARS_RANDLY=true;
    NEARCORR_PROJ=true;
    
    X = cell(L,1);
    P = cell(L,1);
    for l=1:L
        X{l} = rand(k*(k-1)/2,1);
        P{l} = speye(k*(k-1)/2);
    end

    tic
    [alpha_est,rho_est,sigma_rho_est] = argmaxQ(alpha_est,...
                                                rho_est,...
                                                sigma_rho_est, ...
                                                X, ...
                                                P, ...
                                                w, ...
                                                GD_LEARNING_RATE,...
                                                MAX_GD_ITERATIONS, ...
                                                GD_TOLERANCE, ...
                                                INIT_GDVARS_RANDLY, ...
                                                NEARCORR_PROJ);
    time = toc;
    fprintf("Time to run one EM-step: %.2f sec",time);
    verifyLessThan(testCase,time,10);
end
