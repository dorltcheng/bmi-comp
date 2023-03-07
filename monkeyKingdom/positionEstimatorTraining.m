%%% Team Members: WRITE YOUR TEAM MEMBERS' NAMES HERE
%%% BMI Spring 2015 (Update 17th March 2015)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         PLEASE READ BELOW            %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function positionEstimator has to return the x and y coordinates of the
% monkey's hand position for each trial using only data up to that moment
% in time.
% You are free to use the whole trials for training the classifier.

% To evaluate performance we require from you two functions:

% A training function named "positionEstimatorTraining" which takes as
% input the entire (not subsampled) training data set and which returns a
% structure containing the parameters for the positionEstimator function:
% function modelParameters = positionEstimatorTraining(training_data)
% A predictor named "positionEstimator" which takes as input the data
% starting at 1ms and UP TO the timepoint at which you are asked to
% decode the hand position and the model parameters given by your training
% function:

% function [x y] = postitionEstimator(test_data, modelParameters)
% This function will be called iteratively starting with the neuronal data 
% going from 1 to 320 ms, then up to 340ms, 360ms, etc. until 100ms before 
% the end of trial.


% Place the positionEstimator.m and positionEstimatorTraining.m into a
% folder that is named with your official team name.

% Make sure that the output contains only the x and y coordinates of the
% monkey's hand.


function [modelParameters] = positionEstimatorTraining(training_data)
  % Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model
  
  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.
  
  svmTrainDs = {};
  
  numNeuron = size(training_data(1, 1).spikes, 1); % 98
  [numTrial, numDir] = size(training_data); % 80, 8
  t_length = 320;
  
  for dir = 1:numDir
      spikeAvg = zeros(numTrial, numNeuron); % matrix (80 x 98)
      for n = 1:numTrial
          spikeAvg(n, :) = mean(training_data(n, dir).spikes(:, 1:t_length), 2); % time average of each neuron of 1 spike
      end
      svmTrainDs{dir} = spikeAvg;
  end
  
  svmModels = {};
  combos = [1,2,3,4; 5,6,7,8;
            2,3,4,5; 6,7,8,1;
            3,4,5,6; 7,8,1,2;
            4,5,6,7; 8,1,2,3];
  
  for numSvm = 1:4
      dirs_a = combos(2*numSvm - 1, :);
      dirs_b = combos(2*numSvm, :);
      
      svmTrainDs_a = [];
      svmTrainDs_b = [];
      for k = 1:4
          svmTrainDs_a = [svmTrainDs_a, svmTrainDs{dirs_a(k)}'];
          svmTrainDs_b = [svmTrainDs_b, svmTrainDs{dirs_b(k)}'];
      end
      svmTrainDs_a = svmTrainDs_a'; % 4 rows of data from 4 direction classes
      svmTrainDs_b = svmTrainDs_b';
      
      X_train = [svmTrainDs_a; svmTrainDs_b];
      y_train = repelem(0:1, size(training_data,1)*4)'; % 640x1: 320 of 0 and 320 of 1
         
      svm = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', 'Standardize', true, 'KernelScale','auto');
      
%     Or use NBNG's model, but no need since fitcsvm is good enough
%       C = 10; % svm regularization param
%       svmKernel = @linearKernel;
%       tol = 0.01; % tolerance
%       iter = 500; % #iterations before algo quits
%       svm = svmTrain(X_train, y_train, C, svmKernel, tol, iter);

      svmModels{numSvm} = svm;    
  end 
  
  modelParameters.svmModel = svmModels; % saved all 4 trained svm models 
  
  % Kalman filtering training
  
  
  [nb_trials,nb_angles]=size(training_data);


for idx_angle=1:nb_angles
    max_val=1000;
    l=[];
    X_av=zeros(nb_trials,max_val);
    Y_av=zeros(nb_trials,max_val);
    Z_av=zeros(nb_trials,max_val);

    for idx_trial=1:nb_trials
        l=[l, length(training_data(idx_trial,idx_angle).handPos(1,:))];
        X_av(idx_trial,1:l(end))=training_data(idx_trial,idx_angle).handPos(1,:);
        Y_av(idx_trial,1:l(end))=training_data(idx_trial,idx_angle).handPos(2,:);
    end

    duration=min(l);
    Tstart=320;
    Tstop=duration;

    X_av=sum(X_av(:,Tstart:Tstop),1)*(1/nb_trials);
    Y_av=sum(Y_av(:,Tstart:Tstop),1)*(1/nb_trials);
    Z_av=sum(Z_av(:,Tstart:Tstop),1)*(1/nb_trials);

    traj=cat(1,X_av,Y_av,Z_av);

    modelParameters.X{idx_angle}= traj(1,:);
    modelParameters.Y{idx_angle}= traj(2,:);
end
      
        
  

  
  
  
 
  
  
  
end
