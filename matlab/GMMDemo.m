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
  
    %% Plot the data points and their estimated pdfs.

    % Display a scatter plot of the four distributions.
    figure(2);
    hold off;
    plot(X(:, 1), X(:, 2), 'bo');
    hold on;

    set(gcf,'color','white') % White background for the figure.
    
    % First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
    % the input values over the grid.
    gridSize = 100;
    u = linspace(-12, 12, gridSize);
    [A B] = meshgrid(u, u);
    gridX = [A(:), B(:)];

    % Calculate the Gaussian response for every value in the grid.
    z1 = GaussianPDF(gridX, params.means{1}, params.covar{1});
    z2 = GaussianPDF(gridX, params.means{2}, params.covar{2});
    z3 = GaussianPDF(gridX, params.means{3}, params.covar{3});
    z4 = GaussianPDF(gridX, params.means{4}, params.covar{4});

    % Reshape the responses back into a 2D grid to be plotted with contour.
    Z1 = reshape(z1, gridSize, gridSize);
    Z2 = reshape(z2, gridSize, gridSize);
    Z3 = reshape(z3, gridSize, gridSize);
    Z4 = reshape(z4, gridSize, gridSize);

    % Plot the contour lines to show the pdf over the data.
    [C, h] = contour(u, u, Z1);
    [C, h] = contour(u, u, Z2);
    [C, h] = contour(u, u, Z3);
    [C, h] = contour(u, u, Z4);
    axis([0 14 -4 14])

    title('Estimated PDFs with Data');
    
end