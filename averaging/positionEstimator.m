%%% Team Members: Coraline Beitone, Dorothy Cheng, Marco Cheng
%%% METHOD: SVMs + Averaging

function [x,y,newModelParameters] = positionEstimator(test_data,modelParameters)
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
    
%% Averaging estimation
    
    direction = newModelParameters(1).direction;
   
    t_lag = 20;
    k=floor((length(test_data.spikes) - 320)/t_lag);
    
    % Prediction for next steps time
    X = newModelParameters.X{direction};
    Y = newModelParameters.Y{direction};

        if k * t_lag > length(X) 
            V_x = newModelParameters.decodedHandPos_velocity(1);
            V_y = newModelParameters.decodedHandPos_velocity(2);
            
            x_0 = newModelParameters.decodedHandPos_avg(1);
            y_0 = newModelParameters.decodedHandPos_avg(2);
            
            x = V_x * t_lag + x_0;
            y = V_y * t_lag + y_0;
  
        else
            if k==0
                x = test_data.startHandPos(1);
                y = test_data.startHandPos(2);

            else
                x = X(k * t_lag);
                y = Y(k * t_lag);

                if k > 1
                    V_x = abs((X(k*t_lag)-X((k-1)*t_lag))) / t_lag;
                    V_y = abs((Y(k*t_lag)-Y((k-1)*t_lag))) / t_lag;
                    
                    newModelParameters.decodedHandPos_velocity = [V_x; V_y];
                    newModelParameters.decodedHandPos_avg = [x; y];
                end
           
            end
        
        end

end



