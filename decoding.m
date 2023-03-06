% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

% function RMSE = decoding(teamName)
function gtPredSave = decoding(teamName)

load('monkeydata_training.mat');

% Set random number generator
rng(2013);
ix = randperm(length(trial));

addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
train_test_split = 80; % 80:20 train:test
trainingData = trial(ix(1:train_test_split),:);
testData = trial(ix(train_test_split+1:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

% Display trajectory (to be added)
figure
hold on
axis square
grid

% Train Model
modelParameters = positionEstimatorTraining(trainingData);

% Test Model
gtPredSave = zeros(size(testData, 1)*8, 2);

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    count = 0;
    
    for direc=randperm(8) 
        decodedHandPos = [];
        count = count + 1;

        times=320:20:size(testData(tr,direc).spikes,2);
        y_test = direc; % save the correct direction 
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
           
            
%             if nargout('positionEstimator') == 3
%                 [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
%                 modelParameters = newParameters;
%             elseif nargout('positionEstimator') == 2
%                 [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
%             end
%             
%             decodedPos = [decodedPosX; decodedPosY];
%             decodedHandPos = [decodedHandPos decodedPos];
%             
%             meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;

            % for testing the svm only
            newParameters = positionEstimator(past_current_trial, modelParameters);
            modelParameters = newParameters;
        
        y_pred = modelParameters.dirPrediction;
        gtPredSave(8*tr-(8-count), :) = [y_test, y_pred];
%         disp([y_test, y_pred]);
            
        end
%         n_predictions = n_predictions+length(times);
%         hold on
%         plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
%         plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
end

% legend('Decoded Position', 'Actual Position')
% 
% RMSE = sqrt(meanSqError/n_predictions) 
% 
% rmpath(genpath(teamName))

end
