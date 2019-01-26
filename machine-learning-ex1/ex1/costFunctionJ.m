function J = costFunction(X, y, theta)

J = 0;
% X is design matrix
% y is class level

m = size(X,1);
predictions = X*theta;

sqrErrors = (predictions - y) .^ 2;

J = 1/(2*m) * sum(sqrErrors);


