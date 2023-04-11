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
            pred = predict(modelParameters.svmModel{numSvm}, X_test);
            svmPreds(numSvm) = pred;
        end

        % determine the class
        
        %     combos:  [1,2,3,4; 5,6,7,8;
        %               2,3,4,5; 6,7,8,1;
        %               3,4,5,6; 7,8,1,2;
        %               4,5,6,7; 8,1,2,3];
        
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

    %% Linear regression for trajectory prediction
    
    direction = newModelParameters(1).direction;
    selected_neurons = newModelParameters.selectedNeurons; % get selected neurons from training for prediction
    
    t_lag = 20;
    t_min = t_total - t_lag;
    
    % firing rate
    firingRates = zeros(length(selected_neurons), 1);
    for n = 1:length(selected_neurons)
        num_spikes = length(find(test_data.spikes(selected_neurons(n), t_min:t_total)==1));
        firingRates(n) = num_spikes/(t_lag);
    end
    
    % velocity estimation 
    v_x = firingRates' * newModelParameters.regres{direction}(:, 1);
    v_y = firingRates' * newModelParameters.regres{direction}(:, 2);
    
    % trajectory estimation
    if t_total <= 320
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
        
    else % calculate position from velocity 
        x = test_data.decodedHandPos(1,length(test_data.decodedHandPos(1,:))) + v_x * (t_lag);
        y = test_data.decodedHandPos(2,length(test_data.decodedHandPos(2,:))) + v_y * (t_lag);
    end
    
    
end
