
 function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));%as theta=theta-alpha *grad hence dim of theta and grad should be same
n=length(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%copied from previos exercise

z=X*theta;
z=z.*-1;
predict=1./(1+e.^z); %g(z)
temp1=(y'*log(predict)+(1-y')*log(1-predict));

temp2(2:n)=theta(2:n).*theta(2:n);
temp2(1)=0;

J2=(1/(2*m))*lambda*sum(temp2);%factor for regularization 
J1=(-1/m)*sum(temp1);
J=J1+J2% total cost of all training set
temp2=0;
temp1=0;
%now calculating grad using vectorised implementation
%calculate length of theta explicitly inside each case 
%donot use previously calculated size of theta
z=X*theta;
z=z.*-1;
predict=1./(1+e.^z); %g(z)
  temp5=X'*(predict-y);
  temp6=zeros(size(theta));
  temp6(2:(length(theta)))=lambda.*theta(2:length(theta));
  
  grad=(1/m)*(temp5.+temp6);
  
%pasted







% =============================================================

grad = grad(:);

end
