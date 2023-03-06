clear all;
close all;
clc;

%% SVM - for angle classification

% Feature reduction - reducing # neurons?
%   dimensionality reduction

% Steps:
% 1. random select train and test data 
%    - 80 train, 20 test
%    - Can add cross-validation 


% 2. modelParameters = positionEstimatorTraining(train_data)

%    INPUTS:
%    - for every angle: 
%       - each input 'spikes': (80, 98) in shape
%           - each entry is a mean across time 1:320ms of one neuron in each trial 
%    - store in train struct: train{1} = spikes (80, 98) of angle 1

%    SVMS:
%    - 4 sets of combos
%    - for each SVM:
%       - for each combo, take two sets of train data: train_a{1, 2, 3, 4}
%       and train_b{5, 6, 7, 8}
%       - X_train = [train_a, train_b] (shape = (160, 98))
%       - model = fitcsvm()
%    - total models = {model1, model2, model3, model4} (4 SVM in total)
%    - modelParameters.models = models

%    8 KALMAN FILTERING?
%    - modelParameters.filers = filters

% 3. Testing

%    INPUTS: 
%    - test data: past_current_trial 
%       - only use first 1:300ms for the svm to look at

%    SVM PREDICTIONS:
%    - for svm_num = 1:4
%       - pred = predict(modelParameters.svm, X_test)
%       - preds(svm_num) = pred 
%    - prediction combo: 0000, 0001, 0010, 0100 etc.......

%    KALMAN FILTERING

%% 
gtPred = decoding('monkeyKingdom');

accuracy = sum(gtPred(:, 1) == gtPred(:, 2))/size(gtPred, 1);
disp(['Accuracy of direction prediction: ' num2str(accuracy)]);

