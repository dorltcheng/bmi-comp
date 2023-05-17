function preds = SVMPred(model_param, X)

    % Initialize predition vector 
    pred = zeros(size(X, 1), 1);
    preds = zeros(size(X, 1), 1);

    if strcmp(func2str(model_param.kernel), 'linearKernel') % linear kernel for pred output
        pred = X * model_param.w + model_param.b;

    elseif strcmp(func2str(model_param.kernel), 'rbfKernel') % RBF (Gaussian) kernel for pred output
        X_1 = sum(X.^2, 'all');
        X_2 = sum(model_param.X.^2, 2)';

        K = X_1 + X_2 + (-2 * X * model_param.X');
        K = model_param.kernel(1, 0).^ K;
        K = model_param.y'.* K;
        K = model_param.a'.* K;

        pred = sum(K, 2);
    end

    % Binarize the prediction output (1 or 0)
    preds(pred <= 0) = 0;
    preds(pred > 0) = 1;

end

% Reference from: https://github.com/everpeace/ml-class-assignments