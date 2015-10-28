function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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
C_tmp = 0.01;
min_err = 1;
while C_tmp <= 30,
    sigma_tmp = 0.01;
    while sigma_tmp <= 30,
        model = svmTrain(X, y, C_tmp, @(x1, x2) gaussianKernel(x1, x2, sigma_tmp));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval)); %error=fraction incorrect
        if error < min_err,
            min_err = error;
            C = C_tmp;
            sigma = sigma_tmp;
        end;
        fprintf(['error:%f, min_err:%f, C:%f, sigma:%f'], error, min_err, C, sigma);
        sigma_tmp = sigma_tmp * 3;
    end;
    C_tmp = C_tmp * 3;
end;






% =========================================================================

end
