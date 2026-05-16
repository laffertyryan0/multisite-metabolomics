warning('off','all')

%% Set path
addpath('../src')

%% Switch between real or simulated data

USE_REAL_DATA = false;

%% Simulate Data
   
rng(44);

num_metabolites = 50; %k
num_labs = 100; %L
average_fraction_missing_metabolites = 0;
num_mixture_components = 2; %r
mixing_probabilities = ones(1,num_mixture_components)/num_mixture_components;
num_subjects_per_lab = ones(num_labs,1)*6%000; 


% We want to check whether it matters much if the variance is mis-specified
% so generate the correlation matrices from a known variance-mean
% relationship


reported_spearman = {};
reported_spearman_mask = {}; % 1 = non-missing, 0 = missing
n_samples = num_subjects_per_lab;
r = num_mixture_components;
true_rho = {};
true_rho_by_lab = {};
true_Sigrho_by_lab = {};
true_Sigrho_times_nsamples = {};

k = num_metabolites;
L = num_labs;

alpha = mixing_probabilities; % The alpha_j are the probabilities of a lab being 
                   % selected as state j


Gamma = mnrnd(1,alpha,L); % multinomial dist with n=1 is categorical dist
                          % Gamma is Lxr with each row a One Hot Encoded
                          % lab state

for j=1:r
    true_rho{j} = randomCorrelationMatrix(num_metabolites);
    if j==1
        true_rho{j} = eye(num_metabolites);
        true_Sigrho_times_nsamples{j} = 1;%5.27;
    end
    if j==2
        true_rho{j} = .5*eye(num_metabolites) + ones(num_metabolites)*.5;
        true_Sigrho_times_nsamples{j} = .75;%.33;
    end
  %   true_Sigrho_times_nsamples{j} =  5*getCovarianceOfCorrelations(true_rho{j}, ...
   %                                      5-1); % 5 doesn't matter. n-1
                                            % only appears multiplicatively
end


for l=1:num_labs
    lab_state = find(Gamma(l,:));

    p = 1-average_fraction_missing_metabolites; %p = 0 means none are missing
    covariate_mask = rand(1,k) < p; % 1 if non-missing

    reported_spearman_mask{l} = ones(k,k);
    reported_spearman_mask{l}(:,covariate_mask==0)=0;
    reported_spearman_mask{l}(covariate_mask==0,:)=0;

    true_rho_by_lab{l} = true_rho{lab_state};
    true_Sigrho_by_lab{l} = true_Sigrho_times_nsamples{lab_state}/n_samples(l);

    covR = true_Sigrho_by_lab{l};
    covR = (covR + covR')/2;
    corr_vecL = mvnrnd(vecL(true_rho_by_lab{l}), covR);
    reported_spearman{l} = vecLInverse(corr_vecL);

    reported_spearman{l}(reported_spearman_mask{l}==0)=0; 
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

% For each l, find the permutation matrix P_l that puts missing entries 
% below observed entries. Let [X Z]' = P_l * vecL(reported_pearson)
P = {}; 
X = {}; % Consists of all observed entries. 
        % Z here would be represented as all 0's due to the masking, but 
        % really Z means unobserved data. 

for l=1:num_labs
    P{l} = getMaskOrderingMatrix(reported_spearman_mask_vecL{l});
    num_observed = sum(reported_spearman_mask_vecL{l});
    X_Z = P{l}*reported_spearman_vecL{l};
    X{l} = X_Z(1:num_observed);
end

% Scatter plot for debugging
% idx = 1:L;
% pts = cell2mat(X);
% hold on
% scatter(idx(Gamma(:,1)==1),pts(Gamma(:,1)==1),'Color','red');
% scatter(idx(Gamma(:,1)==0),pts(Gamma(:,1)==0),'Color','blue');
% hold off

% calculate (over-labs) sample variances for each rho component
% X_vars = zeros(size(X{1,1},1),1);
% for s=1:size(X_vars,1)
%         X_el = [];
%         for u=1:L
%             X_el = [X_el ; X{u}(s)];
%         end
%             X_vars(s) = var(X_el);
% end

MAX_EM_ITERATIONS = 1000; % Outer loop
MAX_GD_ITERATIONS = Inf; % Inner PGD loop
GD_TOLERANCE = .01;
GD_LEARNING_RATE = 1e-2;%1e-7;%.01*(.2/num_labs)/max(n_samples);
INIT_GDVARS_RANDLY = false;
NEARCORR_PROJ = true; % Do the correlation projection in the gd loop


% Initialize EM parameters
alpha_est = rand(1,r)/2 + 1/4;
alpha_est = alpha_est/sum(alpha_est); % Random initialization
rho_est = cell(1,r);
sigma_rho_est = cell(1,r);

pearson_rho_est = cell(1,r); % Update this on every EM iteration

for j = 1:r
    rho_est{j} = ...
        vecL(randomCorrelationMatrix(num_metabolites)); % Random initialization
    sigma_rho_est{j} = true_Sigrho_times_nsamples{j}; % because the way I 
    % set up the likelihood, the Sigma_rho represents the actual covariance
    % times the sample size. which is why we have the weight factor w
        %sigma_rho_est{j} = 1*speye(num_metabolites*(num_metabolites-1)/2)%+...
       % 1;  
end

w = 1./n_samples; % Lab-wise weighting factor for ...
                                 % variances (L vector)


for em_iter=1:MAX_EM_ITERATIONS
    disp("=========================================================")
    fprintf("EM Iteration Number: %d\n",em_iter);

    % Log current estimate for mixing probabilities
    fprintf("Current alpha estimate: ");
    disp(alpha_est);
    fprintf("\n");

    % Log intermediate spearman correlation matrix calculations
    % and compare with true pearson correlation
    if ~USE_REAL_DATA
        displayMatrixComparison(rho_est,true_rho,4);
    end

    disp('three')
    disp(rho_est)
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

    disp('four')
    disp(rho_est)

    if any(isnan(alpha_est))
        disp("Stopping because of nan values.")
        break
    end
   disp("Finished iteration number: ")
   disp(em_iter)
end

%pearson_rho_est is the final estimate for the mean correlation matrix