% to be filled in

function params = EMMStep(X,K,W)
    N = size(X, 1);
    % For each of the clusters...
    for j = 1 : K
        %================================================================================%
        % Calculate the prior probability for cluster 'j'.
        phi(j) = mean(W(:, j), 1);
        %================================================================================%
        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of all data points.
        % Apply the weights to the values by taking the dot-product between the
        % two vectors.
        weighted_avg = W(:, j)' * X;
        % Divide by the sum of the weights.
        weighted_avg = weighted_avg ./ sum(W(:, j), 1);
        weighted_means(j, :) = weighted_avg;
        %================================================================================%
        % Calculate the covariance matrix for cluster 'j' by taking the 
        % weighted average of the covariance for each training example.
        sigma_k = zeros(2, 2);
        % Subtract the cluster mean from all data points.
        Xm = bsxfun(@minus, X, weighted_means(j, :));
        % Calculate the contribution of each training example to the covariance matrix.
        for i = 1 : N
            sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));
        end
        % Divide by the sum of weights.
        covar{j} = sigma_k ./ sum(W(:, j));
        %================================================================================%
    end
    % Transpose the means for each cluster and store them in the
    % corresponding index as a struct
    for i=1:K
        means{i} = weighted_means(i,:)';
    end
    params.means = means;
    params.covar = covar;
    params.mixCoeff = phi;
end