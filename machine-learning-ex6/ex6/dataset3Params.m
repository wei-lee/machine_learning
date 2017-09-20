function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_list = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30];

m = size(C_list, 2);
n = size(sigma_list, 2);

err_val = -1;

for i = 1:m,
  c_to_use = C_list(i);
  for j = 1:n,  
    sigma_to_use = sigma_list(j);
    model = svmTrain(X, y, c_to_use, @(x1, x2) gaussianKernel(x1, x2, sigma_to_use));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if err_val == -1 || error < err_val,
      err_val = error;
      C = c_to_use;
      sigma = sigma_to_use;     
    endif
  end
end








% =========================================================================

end
