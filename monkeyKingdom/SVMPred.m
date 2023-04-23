function preds = SVMPred(model, X)

% Initialize predition vector 
pred = zeros(size(X, 1), 1);
preds = zeros(size(X, 1), 1);

if strcmp(func2str(model.kernelFun), 'linearKernel') % linear kernel for pred output
    pred = X * model.w + model.b;
    
elseif strcmp(func2str(model.kernelFun), 'rbfKernel') % RBF (Gaussian) kernel for pred output
    X1 = sum(X.^2, 2);
    X2 = sum(model.X.^2, 2)';
    
    K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
    K = model.kernelFun(1, 0) .^ K;
    K = bsxfun(@times, model.y', K);
    K = bsxfun(@times, model.alpha', K);
    
    pred = sum(K, 2);
end

% Binarize the prediction output (1 or 0)
preds(pred <= 0) = 0;
preds(pred > 0) = 1;

end

% Code reference: https://github.com/everpeace/ml-class-assignments