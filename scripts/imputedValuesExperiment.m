%% Set path
addpath('./src')

%% Switch between real or simulated data

USE_REAL_DATA = false;

%% Simulate Data
   

if ~USE_REAL_DATA
    num_metabolites = 10; %k
    num_labs = 60; %L
    average_fraction_missing_metabolites = 0.7;
    num_mixture_components = 1; %r
    mixing_probabilities = ones(1,num_mixture_components)/num_mixture_components;
    num_subjects_per_lab = ones(num_labs,1)*1000; 
    random_seed = 26;    

    rho_state = {};
    for j=1:num_mixture_components
        rho_state{j} = randomCorrelationMatrix(num_metabolites);
        if j==2
            rho_state{j} = eye(num_metabolites,num_metabolites);
        end
        if j==1
           rho_state{j} = patternedBlockCorrelation(num_metabolites, ...
                                                .7, ...
                                                20, ...
                                                20,...
                                                .1);
        end
        assert(min(eig(rho_state{j}))>= 0, ...
            "Non-PSD matrix found for simulation rho_i.")
    end
    % Generate a random mask matrix for lab l
    p = 1-average_fraction_missing_metabolites; %p = 0 means none are missing
    for l=1:num_labs;
        subject_data_mask{l} = rand(1,num_metabolites) < p; % 1 if non-missing
    end

    [reported_spearman,...
              reported_spearman_mask,...
              n_samples,...
              r,...
              true_rho]...
              ...
                  =  simulateData(num_metabolites, ...
                                  num_labs, ...
                                  average_fraction_missing_metabolites, ...
                                  num_mixture_components, ...
                                  subject_data_mask,...
                                  rho_state,...
                                  mixing_probabilities,...
                                  num_subjects_per_lab,...
                                  random_seed...
                                  );
elseif USE_REAL_DATA

    num_mixture_components = 1; %r

    % Read dataset from ./data
    % Each lab is from a seperate CSV file
    data_directory = './data/';

    % Create a master list of uids
    uid_list = {};

    files = dir(fullfile(data_directory,"*.csv"));
    for file = files'
        data = readtable(fullfile(data_directory,file.name));
        uid_list = unique(vertcat(uid_list,data.OUTCOME_UID));
    end

    % Ingest and process the correlation matrix data
    reported_spearman = {};
    reported_spearman_mask = {};
    n_samples = [];
    
    for file = files'        
        disp("Getting data from:")
        disp(file.name)
        data = readtable(fullfile(data_directory,file.name));
        correlation_matrix = zeros(size(uid_list)); 
        mask_matrix = zeros(size(uid_list));
        for i=1:size(uid_list,1)
            for j=1:size(uid_list,1)
                % Find index of (uid_list{i},uid_list{j}) pair
                idx = find(all([strcmp(data.OUTCOME_UID,uid_list{i}) ...
                    strcmp(data.EXPOSURE_UID,uid_list{j})],2));
                if(size(idx,1)==0)
                    % This entry is missing in this lab
                    mask_matrix(i,j) = 0;
                    correlation_matrix(i,j) = 0;
                else
                    if size(idx,1) > 1
                        % Duplicate row. Ignore
                        idx = idx(1);
                    end
                    mask_matrix(i,j) = 1;
                    correlation_matrix(i,j) = data.C_CORR(idx);
                end
            end
        end
        
        reported_spearman{1,end+1} = correlation_matrix;
        reported_spearman_mask{1,end+1} = mask_matrix;
        if size(data,1)>0
            n_samples = [n_samples data.N(1)];
        else
            n_samples = [n_samples 0];
        end
    end

    num_labs = size(files,1);
    r = num_mixture_components;
    num_metabolites = size(uid_list,1);
end

    

%% Using simulated data (lab aggregates only), estimate correlation matrix

% We have access to lab sample sizes, reported_spearman and 
% reported_spearman mask in this section. 

% First we need to place the matrices into vector form 
% The vech function flattens the upper triangular entries of a matrix
% and is a bijection from symmetric matrices to vectors

reported_spearman_vecL = cellfun(@vecL,reported_spearman,...
                                    'UniformOutput',false);
reported_spearman_mask_vecL = cellfun(@vecL,reported_spearman_mask,...
                                        'UniformOutput',false);

% Calculate Pearson using Gaussian assumption
reported_pearson = cellfun(@spearmanToPearson,reported_spearman, ...
                                        'UniformOutput',false);
reported_pearson_vecL = cellfun(@vecL,reported_pearson,...
                                    'UniformOutput',false);


% Calculate Fisher transformed correlation matrices
reported_fisher = cellfun(@atanh,reported_pearson, ...
                                        'UniformOutput',false);
reported_fisher_vecL = cellfun(@vecL,reported_fisher,...
                                    'UniformOutput',false);


% For each l, find the permutation matrix P_l that puts missing entries 
% below observed entries. Let [X Z]' = P_l * vecL(reported_pearson)
P = {}; 
X = {}; % Consists of all observed entries. 
        % Z here would be represented as all 0's due to the masking, but 
        % really Z means unobserved data. 

ntrials = 10;
E_missing_given_known = cell(1,ntrials);

for trial = 1:ntrials

    t = trial/ntrials;
    
    for l=1:num_labs
        P{l} = getMaskOrderingMatrix(reported_spearman_mask_vecL{l});
        num_observed = sum(reported_spearman_mask_vecL{l});
        X_Z = P{l}*reported_fisher_vecL{l};
        X{l} = X_Z(1:num_observed);
    end
    
    
    MAX_EM_ITERATIONS = 5%30; % Outer loop
    MAX_GD_ITERATIONS = 1; % Inner PGD loop
    GD_TOLERANCE = 1;
    GD_LEARNING_RATE = 100*(.2/num_labs)/max(n_samples);
    INIT_GDVARS_RANDLY = true;
    NEARCORR_PROJ = false; % Do the correlation projection in the gd loop
    
    
    % Initialize EM parameters
    alpha_est = rand(1,r);
    alpha_est = alpha_est/sum(alpha_est); % Random initialization
    rho_est = cell(1,r);
    sigma_rho_est = cell(1,r);
    
    pearson_rho_est = cell(1,r); % Update this on every EM iteration
    
    for j = 1:r
        rho_est{j} = ...
            t*vecL(randomCorrelationMatrix(num_metabolites)) + ...
            (1-t)*vecL(eye(num_metabolites,num_metabolites));
        sigma_rho_est{j} = ...
            speye(num_metabolites*(num_metabolites-1)/2);  
    end
    
    w = 1./n_samples; % Lab-wise weighting factor for ...
                                     % variances (L vector)
    
    variance_est = estimateCovOfReportedFisher(reported_fisher, ...
                                                reported_spearman_mask, ...
                                                num_labs, ...
                                                n_samples);
    
    sigma_rho_est{1} = variance_est;
    
    % Metrics to track for plotting. All should have prefix plotvar
    plotvar_mse = {};  %rho mse
    plotvar_bias = {}; %rho bias
    
    
    for em_iter=1:MAX_EM_ITERATIONS
        disp("=========================================================")
        fprintf("EM Iteration Number: %d\n",em_iter);
    
        % Log current estimate for mixing probabilities
        fprintf("Current alpha estimate: ");
        disp(alpha_est);
        fprintf("\n");
    
        % Log intermediate spearman correlation matrix calculations
        % and compare with true pearson correlation
        % Log intermediate spearman correlation matrix calculations
        % and compare with true pearson correlation
    
        if ~USE_REAL_DATA
    
    
            true_rho_fisher = cellfun(@atanh,true_rho, ... % the pearson
                                            'UniformOutput',false);
            rho_est_fisherinv = cellfun(@tanh,rho_est, ... % the pearson
                                            'UniformOutput',false);
            displayMatrixComparison(rho_est_fisherinv,true_rho,4);
    
            [...
             order] = ... % guessed ordering of the estimated rho vectors
                        displayMatrixComparison(rho_est_fisherinv,...
                                        true_rho,4);
        
            % Append new values to plotvar metrics
            for j=1:r
                estimated = rho_est_fisherinv{order(j)};
                actual = vecL(true_rho{j});
                plotvar_mse{j}(em_iter) = norm(estimated-actual,2);
                plotvar_bias{j}(em_iter) = mean(estimated-actual);
            end
        end
    
        % Show current alpha estimate
    
        % Update the EM using the following formula: 
        % theta^{(t+1)} = argmax_{theta_tilde} Q(theta_tilde | theta^{(t)})
        % Here, theta represents the current estimate of parameters:
        % alpha_est: a r-dimensional vector of mixing probabilities
        % rho_est: a cell array of r cells, where rho_est{j} is a 
        %          vector of length k(k-1)/2. (A vectorized correlation matrix)
        % sigma_rho_est: a cell array of r cells, where sigma_est{j} is a 
        %              (k(k-1)/2) x (k(k-1)/2) covariance matrix.
        % For now, we will not estimate sigma_rho_est, but just let it remain 
        % fixed. Later, we will try to show that this is a good approximation.
        % X: The observed data (a cell array where each lab is a cell)
        % P: The cell array of permutation matrices that rearrange X and Z
        % w: lab-wise weighting that multiplies the covariance matrices
        % GD_LEARNING_RATE: The rate parameter for gradient descent
        % MAX_GD_ITERATIONS: Number of iterations for GD in each EM step. This
        %                    can be small, since it is not mandatory for the
        %                    gradient descent to converge in each step
        % GD_TOLERANCE: If the gradient steps fall below this tolerance, the 
        %               gradient descent loop will cease.
    
        [alpha_est,rho_est,sigma_rho_est] = argmaxQFisher(...
                                                    alpha_est,...
                                                    rho_est,...
                                                    sigma_rho_est, ...
                                                    X, ...
                                                    P, ...
                                                    w, ...
                                                    GD_LEARNING_RATE,...
                                                    MAX_GD_ITERATIONS, ...
                                                    GD_TOLERANCE, ...
                                                    INIT_GDVARS_RANDLY, ...
                                                    NEARCORR_PROJ, ...
                                                    em_iter);
    
       
    end
    
    
        CI = getRhoConfidenceIntervals(...
                                        alpha_est,...
                                        rho_est,...
                                        sigma_rho_est, ...
                                        X, ...
                                        P, ...
                                        w, ...
                                        1);
    
            
        L = length(X);
        r = length(alpha_est);
        len_rho = length(rho_est{1}); % = k(k-1)/2
        k = floor((1+sqrt(1+8*len_rho))/2);
        
        % Initialize grad variables randomly
        if(INIT_GDVARS_RANDLY)
            alpha_tilde = rand(1,r);
            alpha_tilde = alpha_tilde/sum(alpha_tilde); % Random initialization
            rho_tilde = cell(1,r);
            sigma_rho_tilde = cell(1,r);
            for j = 1:r
                rho_tilde{j} = vecL(randomCorrelationMatrix(k)); % Random initialization
                sigma_rho_tilde = sigma_rho_est; % This doesn't change
            end
        else
            % Otherwise, initialize to current estimates
            alpha_tilde = alpha_est;
            rho_tilde = rho_est;
            sigma_rho_tilde = sigma_rho_est;
        
        end
        
        
        
        iteration = 1;
        norm_grad = Inf;
        
        % Step 2: Calculate the alpha_tilde that maximizes the alpha_tilde terms
        
        % Calculate Prob( X | Gamma_lj = 1, theta). This is a Lxr matrix
        % This is normal with mean mu = (P_l rho_j)_{1...len(X)} and Sigma = 
        % (P_l Sigma_rho,j P_l')_{1...len(X),1...len(X)}, that is the XX block
        % of the joint (X,Z) distribution 
        pr_x_given_g_abs = zeros(L,r);
        pr_x_given_g_rel = zeros(L,r);
        pr_x_given_g_log = zeros(L,r);
        
        
        % THIS LOOP IS SLOW: FIX 
        mu_Z_given_X = cell(L,r);
        for l=1:L
            % The number of non-missing entries
            x_len = length(X{l});
            % Calculate the mu and sig (permuted vectors) with appropriate
            % partitions for the first mixing component. To be used for 
            % denominators of relative probability ratios
            mu_est_j1 = P{l}*rho_est{1};
            mu_X_est_j1 = mu_est_j1(1:x_len); % Just the non-missing part
            sig_est_j1 = w(l)*P{l}*sigma_rho_est{1}*P{l}'; 
            sig_XX_est_j1 = sig_est_j1(1:x_len,1:x_len); % Just the upper left block
            for j=1:r
                mu_est = P{l}*rho_est{j};
                mu_X_est = mu_est(1:x_len); % Just the non-missing part
                mu_Z_est = mu_est((x_len+1):end); % Just the missing part
                sig_est = w(l)*P{l}*sigma_rho_est{j}*P{l}'; 
                sig_XX_est = sig_est(1:x_len,1:x_len); % Just the upper left block
                sig_XZ_est = sig_est(1:x_len,(x_len+1):end); % Upper right block
                % Calculate conditional mean Z|X, which we will need for rho_tilde
                mu_Z_given_X{l,j} = ...
                        mu_Z_est + sig_XZ_est'*...
                        ((sig_XX_est)\(X{l}-mu_X_est)); %mu_{Z given X} or mu_{2|1}
                % Calculate relative pdf
                pr_x_given_g_rel(l,j) = det(sig_XX_est_j1\sig_XX_est)^(-1/2)*...
                                        exp(-(1/2)*(X{l}-mu_X_est)'*...
                                        ((sig_XX_est)\...
                                        (X{l}-mu_X_est))+...
                                        (1/2)*(X{l}-mu_X_est_j1)'*...
                                        ((sig_XX_est_j1)\...
                                        (X{l}-mu_X_est_j1)));
                %Calculate absolute pdf
                pr_x_given_g_abs(l,j) = (2*pi)^(-x_len/2)*det(sig_XX_est)^(-1/2)*...
                                        exp(-(1/2)*(X{l}-mu_X_est)'*...
                                        ((sig_XX_est)\...
                                        (X{l}-mu_X_est)));
            end
        end
        
        pr_x_given_g_rel(isinf(pr_x_given_g_rel)) = 1e20;
            pr_g_given_x = (pr_x_given_g_rel.*alpha_est)./...
                            sum(pr_x_given_g_rel.*alpha_est,2);
        
        E_missing_given_known{trial} = {};
        for l=1:L
            E_missing_given_known{trial}{l} = P{l}'*[zeros(length(X{l}),1)/0 ; mu_Z_given_X{l}];
        end

end


row = zeros(1,ntrials);
for tr=1:ntrials
    row(tr) = E_missing_given_known{tr}{1}(2)
end
plot(row)