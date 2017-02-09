function [ pdf ] = GaussianPDF(X, mu, Sigma)
% Get the vector length.
n = size(X, 2);
% Subtract the mean from every data point.
meanDiff = bsxfun(@minus, X, mu');
% Calculate the PDF.
pdf = 1 / sqrt((2*pi)^n * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma) .* meanDiff), 2));
end

