% to be filled in

function W = EMEStep(X,K,C)
    %%===============================================
    %% Expectation
    %
    % Calculate the probability for each data point for each distribution.
    N = size(X, 1);
    pdf = zeros(N, K);
    % For each cluster...
    for j = 1 : K
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = GaussianPDF(X, C.means{j}, C.covar{j});
    end
    % Multiply each pdf value by the prior probability for cluster.
    %    pdf  [m  x  k]
    %    phi  [1  x  k]   
    %  pdf_w  [m  x  k]
    pdf_w = bsxfun(@times, pdf, C.mixCoeff(j));
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    %   sum(pdf_w, 2) -- sum over the clusters.
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));

end