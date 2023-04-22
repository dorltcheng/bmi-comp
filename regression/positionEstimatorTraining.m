%%% Team Members: Coraline Beitone, Dorothy Cheng, Marco Cheng

function [modelParameters, firingRates, velocities] = positionEstimatorTraining_regression2(training_data)

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
    classes = [1,2,3,4; 5,6,7,8;
            2,3,4,5; 6,7,8,1;
            3,4,5,6; 7,8,1,2;
            4,5,6,7; 8,1,2,3];

    for numSvm = 1:4
      dirs_a = classes(2*numSvm - 1, :);
      dirs_b = classes(2*numSvm, :);

      svmTrainDs_a = [];
      svmTrainDs_b = [];
      for k = 1:4
          svmTrainDs_a = [svmTrainDs_a, svmTrainDs{dirs_a(k)}']; % 4 rows of data from 4 direction classes
          svmTrainDs_b = [svmTrainDs_b, svmTrainDs{dirs_b(k)}'];
      end
      svmTrainDs_a = svmTrainDs_a'; 
      svmTrainDs_b = svmTrainDs_b';

      X_train = [svmTrainDs_a; svmTrainDs_b];
      
      y_train_a = zeros(size(training_data, 1)*4, 1);
      y_train_b = ones(size(training_data, 1)*4, 1);
      y_train = vertcat(y_train_a, y_train_b); % 640x1 

      svm = fitcsvm(X_train, y_train, 'KernelFunction', 'gaussian', 'Standardize', true, 'KernelScale','auto'); % kernelfunction: 'gaussian' or 'rbf'
      
      svmModels{numSvm} = svm;    
      
    end 

    modelParameters.svmModel = svmModels; % saved all 4 trained svm models 

    
%% Regression
    
% parameters definition
    selected_neurons = 1:98; 
%     selected_neurons = [85,87,89,55,58,41,40,22,27,28,29,3,66,68];
%     selected_neurons = [4,14,18,29,34,36,48,55,67,68,75,77];
%     selected_neurons = [3,7,23,27,28,29,40,41,55,58,61,66,67,68,85,87,88,89,96,98];
    
    t_lag = 20; % time window 20ms
    t_max = 570; % max traj length
    
    firingRates = {};
    velocities = {};
            

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
                    num_spikes = length(find(training_data(tr, dir).spikes(selected_neurons(n), t:t + t_lag)==1));
                    fRate_trial = cat(2, fRate_trial, num_spikes/(t_lag));
                    
                end
                fRates = cat(2, fRates, fRate_trial);
                
            end
            firingRates(n, dir).frate = fRates;
            
            velocities(dir).x = x_Vs;
            velocities(dir).y = y_Vs;
        end
        
    end
            
                
    
%% Linear Regression training
    regres = {};
    
    for dir = 1:numDir
        velocity = [velocities(dir).x; velocities(dir).y]; 
        
        firingR = [];
        for n = 1:length(selected_neurons)
            firingR = cat(1, firingR, firingRates(n, dir).frate);
        end
        
%         disp(num2str(size(velocity)));
%         disp(num2str(size(firingR)));
        
        regres{dir} = lsqminnorm(firingR', velocity');
    end
    
    modelParameters.regres = regres;
    modelParameters.selectedNeurons = selected_neurons;
end

