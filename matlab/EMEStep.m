% to be filled in

function D = EMEStep(X,K,C)
    % Assign a probability for each data point.
    N = size(X, 1);
    pdf = zeros(N, K);
    % For each cluster...
    for j = 1 : K
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = GaussianPDF(X, C.means{j}, C.covar{j});
    end
    % Multiply the pdf by the prior probability for cluster.
    pdf_weighted = bsxfun(@times, pdf, C.mixCoeff(j));
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    D = bsxfun(@rdivide, pdf_weighted, sum(pdf_weighted, 2));
end

function [ pdf ] = GaussianPDF(X, Mean, Sigma)
% Get the vector length.
n = size(X, 2);
% Subtract the mean from every data point.
meanDiff = bsxfun(@minus, X, Mean');
% Calculate the PDF for a multivariate Gaussian.
pdf = 1 / sqrt((2*pi)^n * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma) .* meanDiff), 2));
end