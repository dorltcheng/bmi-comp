%%% Team Members: Coraline Beitone, Dorothy Cheng, Marco Cheng
%%% METHOD: SVMs + Linear Regression

function [modelParameters, firingRates, velocities] = positionEstimatorTraining(training_data)
%% SVM Classifier

    svmTrainDs = {};

    numNeuron = size(training_data(1, 1).spikes, 1); % 98
    [numTrial, numDir] = size(training_data); % 80, 8
    t_length = 320;

    for dir = 1:numDir
      spikeAvg = zeros(numTrial, numNeuron); % matrix (80 x 98)
      for n = 1:numTrial
          spikeAvg(n, :) = mean(training_data(n, dir).spikes(:, 1:t_length), 2); % temporal avg of each neuron of spike for the 320ms prior to motion
      end
      svmTrainDs{dir} = spikeAvg;
    end

    svmModels = {};
    classes = [1,2,3,4; 5,6,7,8;
            2,3,4,5; 6,7,8,1;
            3,4,5,6; 7,8,1,2;
            4,5,6,7; 8,1,2,3];

    for numSvm = 1:4 % for every SVM model: 
      svmTrainDs_0 = [];
      svmTrainDs_1 = [];
        
      dirs_0 = classes(2*numSvm - 1, :);
      dirs_1 = classes(2*numSvm, :);
      
      for k = 1:4
          svmTrainDs_0 = [svmTrainDs_0, svmTrainDs{dirs_0(k)}']; % 4 rows of data from 4 direction classes
          svmTrainDs_1 = [svmTrainDs_1, svmTrainDs{dirs_1(k)}'];
      end
      svmTrainDs_0 = svmTrainDs_0'; 
      svmTrainDs_1 = svmTrainDs_1';

      X_train = [svmTrainDs_0; svmTrainDs_1]; % Training X data
      
      % Target labels (y - 0 and 1)
      y_train_0 = zeros(size(training_data, 1)*4, 1);
      y_train_1 = ones(size(training_data, 1)*4, 1);
      y_train = vertcat(y_train_0, y_train_1); % concatenate target labels 640x1 

      svm = SVM(X_train, y_train, @rbfKernel, 20, 0.01, 500); % fit into SVM function with RBF kernel
      
      svmModels{numSvm} = svm;    
      
    end 

    modelParameters.svmModel = svmModels; % saved all 4 trained svm models 

    
%% Linear Regression
    
    % parameters definition
    selected_neurons = [3,7,23,27,28,29,40,41,55,58,61,66,67,68,85,87,88,89,96,98]; % set of manually selected neurons 
    
    t_lag = 20; % time window 20ms
    t_max = 570; % max traj length
            
    for dir = 1:numDir % for every direction
        x_Vs = []; % save all x velocities in one direction for all neurons
        y_Vs = []; % save all y velocities in one direction for all neurons

        for n = 1:length(selected_neurons) % for every neuron
            fRates = []; % store all firing rates for one neuron over all trials
            
            for tr = 1:numTrial % for every trial 
                fRate_trial = []; % save the firing rate of one single trial
                
                for t = 320:t_lag:t_max - t_lag % for every time bin (20ms)
                    
                    % Calculate velocity once only for every trial  
                    if n == 1
                        x = training_data(tr, dir).handPos(1, t);
                        x_next = training_data(tr, dir).handPos(1, t + t_lag);
                        vel_x = (x_next - x)/(t_lag);
                        
                        y = training_data(tr, dir).handPos(2, t);
                        y_next = training_data(tr, dir).handPos(2, t + t_lag);
                        vel_y = (y_next - y)/(t_lag);
                        
                        x_Vs = cat(2, x_Vs, vel_x);
                        y_Vs = cat(2, y_Vs, vel_y);
                    end
                    
                    % Calculate spike rate at each time bin
                    with_spikes = find(training_data(tr, dir).spikes(selected_neurons(n), t:t + t_lag)==1);
                    fRate_trial = cat(2, fRate_trial, length(with_spikes)/(t_lag));
                    
                end
                fRates = cat(2, fRates, fRate_trial);
                
            end
            firingRates(n, dir).frate = fRates;
            
            velocities(dir).x = x_Vs;
            velocities(dir).y = y_Vs;
        end
        
    end
            
    
%% Linear Regression training

    % Function for minimum norm least-squares method (a linear regressor) 
    function x = least_squares_minnorm(A, b)
        
        [U, S, V] = svd(A, 'econ'); % SVD of matrix A
        inv_S = diag(1 ./ diag(S)); % reciprocal of non-zero singular values
        x = V * inv_S * U' * b; % min-norm LS solution
        
    end

    % Linear regression training for each direction (8 regressors in total)
    regres = {};
    
    for dir = 1:numDir
        velocity = [velocities(dir).x; velocities(dir).y]; 
        
        firingR = [];
        for n = 1:length(selected_neurons)
            firingR = cat(1, firingR, firingRates(n, dir).frate);
        end
        
        regres{dir} = least_squares_minnorm(firingR', velocity');
        
    end
    
    % saving model parameters
    modelParameters.regres = regres;
    modelParameters.selectedNeurons = selected_neurons;
end

