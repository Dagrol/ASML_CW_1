% to be filled in

function params = GMMInit(X,K)
    % Set 'N' to the number of data points.
    N = size(X, 1);
    % Randomly select k data points to serve as the initial means.
    indeces = randperm(N);
    for i = 1:K
        means{i} = X(indeces(i),:)';
    end
    % Use the overal covariance of the dataset as the initial variance for each cluster.
    for j = 1 : K
        covars{j} = cov(X);
    end
    % Assign equal prior probabilities to each cluster.
    phis = ones(1, K) * (1 / K);
    params.means = means;
    params.covar = covars;
    params.mixCoeff = phis;
end