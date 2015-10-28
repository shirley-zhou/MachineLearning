function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

all_combos = eye(num_labels);    
Y = all_combos(y,:);         %creates a logical matrix out of vector y
                            %by picking out and rearranging the correct row
                            %of identity matrix; Y is m x num_labels

A1 = [ones(m,1) X]; %A1 is m x num_labels+1 = 5000 x 401
A2 = sigmoid(Theta1*A1'); %A2 is 25 x 5000
A2 = [ones(1, size(A2, 2)); A2]; %A2 is 26 x 5000
H = sigmoid(Theta2*A2); %H is 10 x 5000
%for k=1:num_labels,
%   J = J + 1/m * (-log(H(k,:))*Y(:,k) - log(1-H(k,:))*(1-Y(:,k)));
%end;

%vectorized version:
cost = -Y.*log(H)' - (1-Y).*log(1-H)'; %5000 x 10 matrix of indiv costs for each y(i,k)
J = 1/m * sum(cost(:)); %use (:) to concat to vector and add elements
                        %otherwise sum adds over cols of matrix
%regularize:
Theta1Sqr = Theta1(:,2:end).^2;
Theta2Sqr = Theta2(:,2:end).^2;
J = J + lambda/(2*m) * ( sum(Theta1Sqr(:)) + sum(Theta2Sqr(:)) );

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for t=1:m,
    a1 = [1 X(t,:)];
    z2 = Theta1*a1';
    a2 = [1; sigmoid(z2)];
    z3 = Theta2*a2;
    h = sigmoid(z3);
    
    d3 = h - Y(t,:)';
    d2 = Theta2(:,2:end)'*d3 .* sigmoidGradient(z2);
    D2 = D2 + d3*a2';
    D1 = D1 + d2*a1; 
end;

Theta1_grad = D1/m;
Theta2_grad = D2/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
