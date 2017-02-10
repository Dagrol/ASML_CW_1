% Advanced Statistical Machine Learning & Pattern Recognition - CO495
% skeleton for CW1

function params = GMMDemo()
    load('X.mat','X'); % load data
    K = 4; % number of Gaussians in the GMM

    params = GMMInit(X,K); % initialise the parameters
    % params should be a struct with fields
    % means: { [2x1 double] [2x1 double] [2x1 double] [2x1 double] }
    % covar: { [2x2 double] [2x2 double] [2x2 double] [2x2 double] }
    % mixCoeff: { [1x1 double] [1x1 double] [1x1 double] [1x1 double] }
    
    % EM algorithm
    for i = 1:10 % do not change --- keep the iterations fixed
        % E step
        resp = EMEStep(X,K,params); % compute responsibilities, i.e., every \gamma(z_{n,k}), size(resp) = [size(X,1),K]
        
        % M step
        params = EMMStep(X,K,resp); % update the values for the parameters
        
    end
    writetable(struct2table(params), 'params.xlsx') % save struct params as a spreadsheet 
    
end