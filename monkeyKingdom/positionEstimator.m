
% function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
function [newModelParameters] = positionEstimator(test_data, modelParameters)

    % **********************************************************
    %
    % You can also use the following function header to keep your state
    % from the last iteration
    %
    % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
    %                 ^^^^^^^^^^^^^^^^^^
    % Please note that this is optional. You can still use the old function
    % declaration without returning new model parameters. 
    %
    % *********************************************************

    % - test_data:
    %     test_data(m).trialID
    %         unique trial ID
    %     test_data(m).startHandPos
    %         2x1 vector giving the [x y] position of the hand at the start
    %         of the trial
    %     test_data(m).decodedHandPos
    %         [2xN] vector giving the hand position estimated by your
    %         algorithm during the previous iterations. In this case, N is 
    %         the number of times your function has been called previously on
    %         the same data sequence.
    %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
    %     in this case, t goes from 1 to the current time in steps of 20
    %     Example:
    %         Iteration 1 (t = 320):
    %             test_data.trialID = 1;
    %             test_data.startHandPos = [0; 0]
    %             test_data.decodedHandPos = []
    %             test_data.spikes = 98x320 matrix of spiking activity
    %         Iteration 2 (t = 340):
    %             test_data.trialID = 1;
    %             test_data.startHandPos = [0; 0]
    %             test_data.decodedHandPos = [2.3; 1.5]
    %             test_data.spikes = 98x340 matrix of spiking activity



    % ... compute position at the given timestep.

    % Return Value:

    % - [x, y]:
    %     current position of the hand

    newModelParameters = modelParameters;
    t_length = 300; % look at first 300ms for svm prediction
    X_test = mean(test_data.spikes(:, 1:t_length), 2);
    t_total = size(test_data.spikes, 2); % total length of spike test data
    X_test = X_test';
    
    %     combos = [1,2,3,4; 5,6,7,8;
    %               2,3,4,5; 6,7,8,1;
    %               3,4,5,6; 7,8,1,2;
    %               4,5,6,7; 8,1,2,3];
    
    % if test_data.spike is the first sample from each spike trial (with length 320ms, 
    % need to pass into svm to predict direction)
    if t_total ~= 320
        y_pred = modelParameters.dirPrediction;
        if y_pred == 0
            disp('Prediction error (0)')
        end
        
    else % if t_total == 320 (need svm)
        svmPreds = zeros(4, 1);
        for numSvm = 1:4
            pred = predict(modelParameters.svmModel{numSvm}, X_test);
            
%       Or use NBNG's model but no need:
%             pred = svmPredict(modelParameters.svmModel{numSvm}, X_test); % y_pred = 0 or 1

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
                        disp('Wrong pred: 2 or 4');
                    else
                        y_pred = 2; % 0011
                    end
                end
            else
                if svmPreds(3) == 0 % 010_
                    if svmPred(4) == 0 
                        opts = [4, 6]; % 4 or 6
                        y_pred = opts(randi([1, 2], 1)); % 0100
                        disp('Wrong pred: 4 or 6');
                    else
                        opts = [1, 3] % 1 or 3
                        y_pred = opts(randi([1, 2], 1)) % 0101
                        disp('Wrong pred: 1 or 3');
                    end
                else % 011_
                    if svmPreds(4) == 0 
                        opts = [1, 7]; % 1 or 7
                        y_pred = opts(randi([1, 2], 1)); % 0110
                        disp('Wrong pred: 1 or 7')
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
                        disp('Wrong pred: 3 or 5');
                    end
                else % 101_
                    if svmPreds(4) == 0
                        opts = [5, 7] % 5 or 7
                        y_pred = opts(randi([1, 2], 1)); % 1010
                        disp('Wrong pred: 5 or 7');
                    else
                        opts = [2, 8]
                        y_pred = opts(randi([1, 2], 1)); % 1011
                        disp('Wrong pred: 2 or 8');
                    end
                end
            else % 11__
                if svmPreds(3) == 0
                    if svmPreds(4) == 0
                        y_pred = 6; % 1100
                    else
                        opts = [6, 8]; % 6 or 8
                        y_pred = opts(randi([1, 2], 1)); % 1101
                        disp('Wrong pred: 6 or 8')
                    end
                else
                    if svmPreds(4) == 0
                        y_pred = 7; % 1110
                    else
                        y_pred =8; % 1111
                    end
                end
            end
        end
        newModelParameters.dirPrediction = y_pred;
    end

    % Kalman filtering (from y_pred)
    
    
    
    
    
    
    
end
