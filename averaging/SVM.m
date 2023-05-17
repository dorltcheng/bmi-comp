function svm_model = SVM(X, Y, kernel, C, tol, max_iteration)
    
    % X - training X
    % Y - training Y (labels)
    % kernel - linear or Gaussian kernel for non-linearly separable
    % data transformation
    % C - regularization parameter of SVM 
    % tol - tolerance to determine equality of floating point numbers
    % max_iteration - maximum number of iterations 
    

    Y(Y==0) = -1; % class label change from 0 to -1
    
    % Define parameters
    a = zeros(size(X, 1), 1);
    E = zeros(size(X, 1), 1);
    b = 0;
    iter = 0;

    % Compute the kernel function
    if strcmp(func2str(kernel), 'linearKernel') % Linear kernel
        K = X * X';
    elseif strcmp(func2str(kernel), 'rbfKernel') % RBF (Gaussian) Kernel
        X2 = sum(X.^2, 2);
        K = X2 + X2' - 2 * (X * X');
        K = kernel(1, 0) .^ K;
    end

    % Training
    while iter < max_iteration
        change_a = 0;
        for i = 1:size(X, 1)

            % Compute Ei
            E(i) = b + sum(a.*Y.*K(:,i)) - Y(i);

            if ((Y(i)*E(i) < -tol && a(i) < C) || (Y(i)*E(i) > tol && a(i) > 0))

                % select j randomly for simplicity 
                j = ceil(size(X, 1) * rand());
                while j == i
                    j = ceil(size(X, 1) * rand());
                end

                % Compute Ej
                E(j) = b + sum(a.*Y.*K(:,j)) - Y(j);
                
                % Saving old a
                old_a_i = a(i);
                old_a_j = a(j);

                % Compute L and H
                if (Y(i) == Y(j))
                    L = max(0, a(j) + a(i) - C);
                    H = min(C, a(j) + a(i));
                else
                    L = max(0, a(j) - a(i));
                    H = min(C, C + a(j) - a(i));
                end

                % Compute new value for a j
                a(j) = a(j) - (Y(j) * (E(i) - E(j))) / (2 * K(i,j) - K(i,i) - K(j,j));

                % Clipping a
                a(j) = max(L, a(j));
                a(j) = min(H, a(j));
                
                if (abs(a(j) - old_a_j) < tol)
                    a(j) = old_a_j;
                    continue;
                end

                % Compute value for a i 
                a(i) = a(i) + Y(i) * Y(j) * (old_a_j - a(j));

                % Compute b1 and b2
                b1 = b - E(i) - Y(i) * (a(i) - old_a_i) *  K(i,j)' - Y(j) * (a(j) - old_a_j) *  K(i,j)';
                b2 = b - E(j) - Y(i) * (a(i) - old_a_i) *  K(i,j)' - Y(j) * (a(j) - old_a_j) *  K(j,j)';

                % Compute b 
                if (0 < a(j) && a(j) < C)
                    b = b2;
                elseif (0 < a(i) && a(i) < C)
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
    idx = a > 0;
    svm_model.y = Y(idx);
    svm_model.X = X(idx,:);
    svm_model.b = b;
    svm_model.a = a(idx);
    svm_model.w = ((a.*Y)'*X)';
    svm_model.kernel = kernel;

end

% Reference from: https://github.com/everpeace/ml-class-assignments