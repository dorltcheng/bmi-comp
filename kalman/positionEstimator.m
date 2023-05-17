%%% Team Members: Coraline Beitone, Dorothy Cheng, Marco Cheng
%%% METHOD: SVMs + Kalman Filtering

function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
    newModelParameters = modelParameters; % use newModelParameters to update below
    
%% SVM Prediction of Direction
    
    t_length = 320; % look at first 320ms for svm prediction
    X_test = mean(test_data.spikes(:, 1:t_length), 2);
    X_test = X_test';
    t_total = size(test_data.spikes, 2); % total length of spike test data
    
    % if test_data.spike is the first sample from each spike trial (with length 320ms, 
    % need to pass into svm to predict direction)
    if t_total ~= 320
        y_pred = modelParameters.direction;
        
    else % if t_total == 320 (need svm)
        svm_p = zeros(4, 1);
        for numSvm = 1:4
            pred = SVMPred(modelParameters.svmModel{numSvm}, X_test);
            svm_p(numSvm) = pred;
        end

        % determine the class
                
        if svm_p(1) == 0
            if svm_p(2) == 0
                if svm_p(3) == 0
                    if svm_p(4) == 0
                        y_pred = 4; % 0000
                    else
                        y_pred = 3; % 0001
                    end
                else
                    if svm_p(4) == 0
                        opts = [2, 4]; % 2 or 4 
                        y_pred = opts(randi([1, 2], 1)); % 0010 - random assign
                    else
                        y_pred = 2; % 0011
                    end
                end
            else
                if svm_p(3) == 0 % 010_
                    if svmPred(4) == 0 
                        opts = [4, 6]; % 4 or 6
                        y_pred = opts(randi([1, 2], 1)); % 0100 - random assign
                    else
                        opts = [1, 3]; % 1 or 3
                        y_pred = opts(randi([1, 2], 1)); % 0101 - random assign
                    end
                else % 011_
                    if svm_p(4) == 0 
                        opts = [1, 7]; % 1 or 7
                        y_pred = opts(randi([1, 2], 1)); % 0110 - random assign
                    else
                        y_pred = 1; % 0111
                    end
                end
            end

        else % 1___
            if svm_p(2) == 0
                if svm_p(3) == 0
                    if svm_p(4) == 0 
                        y_pred = 5; % 1000
                    else
                        opts = [3, 5]; % 3 or 5
                        y_pred = opts(randi([1, 2], 1)); % 1001 - random assign
                    end
                else % 101_
                    if svm_p(4) == 0
                        opts = [5, 7]; % 5 or 7
                        y_pred = opts(randi([1, 2], 1)); % 1010 - random assign
                    else
                        opts = [2, 8];
                        y_pred = opts(randi([1, 2], 1)); % 1011 - random assign
                    end
                end
            else % 11__
                if svm_p(3) == 0
                    if svm_p(4) == 0
                        y_pred = 6; % 1100
                    else
                        opts = [6, 8]; % 6 or 8
                        y_pred = opts(randi([1, 2], 1)); % 1101 - random assign
                    end
                else
                    if svm_p(4) == 0
                        y_pred = 7; % 1110
                    else
                        y_pred = 8; % 1111
                    end
                end
            end
        end
        newModelParameters.direction = y_pred;
    end

%% Kalman Filtering for (x,y) prediction
    
    predicted_angle = newModelParameters(1).direction;
    
    lag = 20; % time bin for estimation
    nb_states = 4; % (x,y,Vx,Vy) states coordinates 
    % NB: testing phase revealed that the inclusion of acceleration components in the state vector did not improved the performance of the decoder. 
    % Therefore, acceleration was excluded (only 4 states are used).
    Tstart = 320; %  prediction starts at 320 ms 
    I = eye(nb_states); 


    % get Kalman filter parameters for predicted angle
    H = newModelParameters.H{predicted_angle};
    Q = newModelParameters.Q{predicted_angle};
    A = newModelParameters.A{predicted_angle};
    W = newModelParameters.W{predicted_angle};
    selected_neurons = modelParameters.selected_neurons;

    
    % Compute firing rate 
    zk = test_data.spikes(selected_neurons,Tstart+1:end);

    if isempty(zk) % t=320 ms (no motion)
        zk = [];
    else %(motion starts)
        zk = zk(:,end-lag+1:end); % only keep the last 20 ms for the prediction 
        zk = (sum(zk,2)/lag); % firing rate 
       
    end
    
    %  Kalman filter initialization
    if isempty(zk)
        prior = zeros(nb_states); 
        xk = zeros(nb_states,1);
        xk(1:2) = test_data.startHandPos; % first estimate: actual initial position
        x = xk(1);
        y = xk(2);
        Kk = prior*H'/(H*prior*H'+Q);
        newModelParameters.Kk = Kk;
        newModelParameters.posterior = prior;
        newModelParameters.decodedHandPos = xk;

    % Prediction for next time steps
    else
        xk_previous_estimate = newModelParameters.decodedHandPos;
        posterior = newModelParameters.posterior;
        prior = A*posterior*A'+W;

        % Kalman filter parameters update
        % Reference: W. Wu, M. Black, Y. Gao, E. Bienenstock, M. Serruya,
        % and J. Donoghue, "Inferring hand motion from multi-cell
        % recordings in motor cortex using a kalman filter" (2002)

        Kk = prior*H'/(H*prior*H'+Q);
        xk_estimate = A*xk_previous_estimate;
        xk = xk_estimate+Kk*(zk-H*xk_estimate);
        newModelParameters.posterior = (I-Kk*H)*prior;
        newModelParameters.decodedHandPos = xk;
        x = xk(1);
        y = xk(2);
    end

    
end
