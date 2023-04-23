function svm_model = SVM(X, Y, kernelFun, C, tol, max_iteration)
    
    % X - training X
    % Y - training Y (labels)
    % kernelFun - linear or Gaussian kernel for non-linearly separable
    % data transformation
    % C - regularization parameter of SVM 
    % tol - tolerance to determine equality of floating point numbers
    % max_iteration - maximum number of iterations 
    

    Y(Y==0) = -1; % class label change from 0 to -1
    
    % Define parameters
    alpha = zeros(size(X, 1), 1);
    E = zeros(size(X, 1), 1);
    b = 0;
    iter = 0;

    % Compute the kernel function
    if strcmp(func2str(kernelFun), 'linearKernel') % Linear kernel
        K = X * X';
    elseif strcmp(func2str(kernelFun), 'rbfKernel') % RBF (Gaussian) Kernel
        X2 = sum(X.^2, 2);
        K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
        K = kernelFun(1, 0) .^ K;
    end

    % Training
    while iter < max_iteration
        change_a = 0;
        for i = 1:size(X, 1)

            % Compute Ei
            E(i) = b + sum(alpha.*Y.*K(:,i)) - Y(i);

            if ((Y(i)*E(i) < -tol && alpha(i) < C) || (Y(i)*E(i) > tol && alpha(i) > 0))

                % select j randomly for simplicity 
                j = ceil(size(X, 1) * rand());
                while j == i
                    j = ceil(size(X, 1) * rand());
                end

                % Compute Ej
                E(j) = b + sum(alpha.*Y.*K(:,j)) - Y(j);
                
                % Saving old alpha
                old_alpha_i = alpha(i);
                old_alpha_j = alpha(j);

                % Compute L and H
                if (Y(i) == Y(j))
                    L = max(0, alpha(j) + alpha(i) - C);
                    H = min(C, alpha(j) + alpha(i));
                else
                    L = max(0, alpha(j) - alpha(i));
                    H = min(C, C + alpha(j) - alpha(i));
                end

                % Compute new value for alpha j
                alpha(j) = alpha(j) - (Y(j) * (E(i) - E(j))) / (2 * K(i,j) - K(i,i) - K(j,j));

                % Clipping alpha
                alpha(j) = max(L, alpha(j));
                alpha(j) = min(H, alpha(j));
                
                if (abs(alpha(j) - old_alpha_j) < tol)
                    alpha(j) = old_alpha_j;
                    continue;
                end

                % Compute value for alpha i 
                alpha(i) = alpha(i) + Y(i) * Y(j) * (old_alpha_j - alpha(j));

                % Compute b1 and b2
                b1 = b - E(i) - Y(i) * (alpha(i) - old_alpha_i) *  K(i,j)' - Y(j) * (alpha(j) - old_alpha_j) *  K(i,j)';
                b2 = b - E(j) - Y(i) * (alpha(i) - old_alpha_i) *  K(i,j)' - Y(j) * (alpha(j) - old_alpha_j) *  K(j,j)';

                % Compute b 
                if (0 < alpha(j) && alpha(j) < C)
                    b = b2;
                elseif (0 < alpha(i) && alpha(i) < C)
                    b = b1;
                end

                change_a = change_a + 1;

            end

        end

        % update #iteration 
        if (change_a == 0)
            iter = iter + 1;
        else
            iter = 0;
        end

    end
    
    % Saving model parameters
    idx = alpha > 0;
    svm_model.X = X(idx,:);
    svm_model.y = Y(idx);
    svm_model.b = b;
    svm_model.alpha = alpha(idx);
    svm_model.w = ((alpha.*Y)'*X)';
    svm_model.kernelFun = kernelFun;

end

% Code reference: https://github.com/everpeace/ml-class-assignments