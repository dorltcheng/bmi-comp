function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
    newModelParameters = modelParameters; % use newModelParameters to update below
    
    %% SVM Prediction of Direction
    
    t_length = 300; % look at first 300ms for svm prediction
    X_test = mean(test_data.spikes(:, 1:t_length), 2);
    X_test = X_test';
    t_total = size(test_data.spikes, 2); % total length of spike test data
    
    % if test_data.spike is the first sample from each spike trial (with length 320ms, 
    % need to pass into svm to predict direction)
    if t_total ~= 320
        y_pred = modelParameters.direction;
        if y_pred == 0
            disp('Prediction error (0)')
        end
        
    else % if t_total == 320 (need svm)
        svmPreds = zeros(4, 1);
        for numSvm = 1:4
            pred = SVMPred(modelParameters.svmModel{numSvm}, X_test);
            svmPreds(numSvm) = pred;
        end

        % determine the class
                
        if svmPreds(1) == 0
            if svmPreds(2) == 0
                if svmPreds(3) == 0
                    if svmPreds(4) == 0
                        y_pred = 4; % 0000
                    else
                        y_pred = 3; % 0001
                    end
                else
                    if svmPreds(4) == 0
                        opts = [2, 4]; % 2 or 4 
                        y_pred = opts(randi([1, 2], 1)); % 0010
                    else
                        y_pred = 2; % 0011
                    end
                end
            else
                if svmPreds(3) == 0 % 010_
                    if svmPred(4) == 0 
                        opts = [4, 6]; % 4 or 6
                        y_pred = opts(randi([1, 2], 1)); % 0100
                    else
                        opts = [1, 3]; % 1 or 3
                        y_pred = opts(randi([1, 2], 1)); % 0101
                    end
                else % 011_
                    if svmPreds(4) == 0 
                        opts = [1, 7]; % 1 or 7
                        y_pred = opts(randi([1, 2], 1)); % 0110
                    else
                        y_pred = 1; % 0111
                    end
                end
            end

        else % 1___
            if svmPreds(2) == 0
                if svmPreds(3) == 0
                    if svmPreds(4) == 0 
                        y_pred = 5; % 1000
                    else
                        opts = [3, 5]; % 3 or 5
                        y_pred = opts(randi([1, 2], 1)); % 1001
                    end
                else % 101_
                    if svmPreds(4) == 0
                        opts = [5, 7]; % 5 or 7
                        y_pred = opts(randi([1, 2], 1)); % 1010
                    else
                        opts = [2, 8];
                        y_pred = opts(randi([1, 2], 1)); % 1011
                    end
                end
            else % 11__
                if svmPreds(3) == 0
                    if svmPreds(4) == 0
                        y_pred = 6; % 1100
                    else
                        opts = [6, 8]; % 6 or 8
                        y_pred = opts(randi([1, 2], 1)); % 1101
                    end
                else
                    if svmPreds(4) == 0
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
    
    direction = newModelParameters(1).direction;
    selected_neurons = newModelParameters.selectedNeurons; % get selected neurons from training for prediction
    
    % initialise if incoming test_data is 320 in length (start of the data)
    if t_total == 320 
        newModelParameters.P{direction} = zeros(newModelParameters.state_num);
        
        start_states = mean(modelParameters.start_states{direction}, 2);
        start_states(1:2) = test_data.startHandPos; % replace start_states (x, y) with test_data.startHandPos
        newModelParameters.state{direction} = start_states;
    end
    
    % Get trained matrices for the predicted direction 
    A = newModelParameters.A{direction};
    W = newModelParameters.W{direction};
    H = newModelParameters.H{direction};
    Q = newModelParameters.Q{direction};
    Mz = newModelParameters.Mz{direction}; 
    Mx = newModelParameters.Mx{direction};
    P_prev = newModelParameters.P{direction};
    
    prev_state = newModelParameters.state{direction}; % get previous state saved
 
    end_t = size(test_data.spikes, 2); 
    
    t_lag = 20;
    frate = sum(test_data.spikes(selected_neurons, (end_t - t_lag + 1):end_t), 2) / t_lag;
    z_test = frate - Mz;
    
    ap_state = A * prev_state; % a priori estimate 
    P_ap = A * P_prev * A' + W; % error covariance matrix of ap_state
    K = P_ap * H' / (H * P_ap * H' + Q); % Kalman Gain
    
    % Compute new state (x,y)
    new_state = ap_state + K * (z_test - H * (ap_state - Mx));
    x = new_state(1);
    y = new_state(2);
    
    P = (eye(newModelParameters.state_num) - K * H) * P_ap; % update P with neural data and K
    
    % update model parameters
    newModelParameters.P{direction} = P;
    newModelParameters.state{direction} = new_state;

    
end
