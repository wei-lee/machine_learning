function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

r_theta = theta(2:end);

J = 1/m * (-y' * log(h) - (1-y)'*log(1 - h)) + lambda/(2*m)*(r_theta' * r_theta);

for i = 1:size(theta)
  current_theta = theta(i);
  current_grad = 0;
  
  for j = 1:m
    h1 = h(j); % = sigmoid(X(j,:) * theta)
    current_grad = current_grad + (h1 - y(j,1))*X(j,i);
  end
  
  if i == 1
    grad(i) = 1/m*current_grad;
  else
    grad(i) = 1/m*current_grad + lambda/m * current_theta;
  end
end


% =============================================================

end
