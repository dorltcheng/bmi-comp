%%% Team Members: Coraline Beitone, Dorothy Cheng, Marco Cheng

function [modelParameters, firingRates, velocities] = positionEstimatorTraining(training_data)

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

    
%% Linear Regression
    
% parameters definition
    selected_neurons = 1:98; % work best for regression 
%     selected_neurons = [85,87,89,55,58,41,40,22,27,28,29,3,66,68];
%     selected_neurons = [4,14,18,29,34,36,48,55,67,68,75,77];
%     selected_neurons = [3,7,23,27,28,29,40,41,55,58,61,66,67,68,85,87,88,89,96,98];
    
    t_lag = 20; % time window 20ms
    t_max = 550;


  % Get velocity and firing rates
    firingRates = struct([]);
    velocities = struct([]);

    for dir = 1:numDir
        xVel = [];
        yVel = [];

        for n = 1:length(selected_neurons)
            fRates = [];
            for tr = 1:numTrial
                spikeR = [];
                for t = 320:t_lag:t_max - t_lag
                    num_spikes = length(find(training_data(tr, dir).spikes(selected_neurons(n), t:t+t_lag)==1));
                    spikeR = cat(2, spikeR, num_spikes/(t_lag));
                    
                    if n == 1
                        x_t = training_data(tr, dir).handPos(1, t);
                        x_dt = training_data(tr, dir).handPos(1, t+t_lag);
                        y_t = training_data(tr, dir).handPos(2, t);
                        y_dt = training_data(tr, dir).handPos(2, t+t_lag);
                        
                        vel_x = (x_dt - x_t)/(t_lag);
                        vel_y = (y_dt - y_t)/(t_lag);
                        xVel = cat(2, xVel, vel_x);
                        yVel = cat(2, yVel, vel_y);
                    end
                end
                fRates = cat(2, fRates, spikeR);
                
            end
            firingRates(n, dir).frate = fRates;
            velocities(dir).x = xVel;
            velocities(dir).y = yVel;
            
            
        end
      
    end
            
    
% Linear Regression with least square method 
    regres = {};
    
    for dir = 1:numDir
        vel = [velocities(dir).x; velocities(dir).y];
        
        firingR = [];
        for n = 1:length(selected_neurons)
            firingR = cat(1, firingR, firingRates(n, dir).frate);
        end
        
        disp(num2str(size(vel)));
        disp(num2str(size(firingR)));
        
        regres{dir} = lsqminnorm(firingR', vel');
    end
    
    modelParameters.regres = regres;
    modelParameters.selectedNeurons = selected_neurons;
end



%% wrong codes
%     function [vel] = get_vel(trainData, t_lag, nTrial, nDir)
%         length_spikes = 550;
%         numBin = floor((length_spikes - 320) / t_lag);
%         
%         pos = zeros(2, numBin + 1); % for x,y
%         vel = zeros(2, numBin + 1);
%         
%         % x, y, v_x, v_y
%         bin = 320 + (0: t_lag: numBin*t_lag);
%         pos(1, :) = trainData(nTrial, nDir).handPos(1, bin); % x
%         pos(2, :) = trainData(nTrial, nDir).handPos(2, bin); % y
%         
%         vel(1, 1:numBin) = diff(pos(1, :)) / t_lag; % v_x
%         vel(2, 1:numBin) = diff(pos(2, :)) / t_lag; % v_y
%         
%         vel = vel(:, 1:numBin);
%     end
% 
%     function [frate] = get_firingRate(trainData, neuron, t_lag, nTrial, nDir)
%         length_spike = 550;    
% %         firingRate = [];
%         
%         frate = [];
%         for t = 320:t_lag:length_spike - t_lag
%             num_spikes = length(find(trainData(nTrial, nDir).spikes(neuron, t:t+t_lag)==1));
%             frate = cat(2, frate, num_spikes/(t_lag*0.001));
%         end
% %         firingRate = cat(2, firingRate, frate); 
%         
%     end
% 
% 
%     firingRates = struct([]);
%     velocities = struct([]);
%     
%     for dir = 1:numDir
%         vels = [];
%         for tr = 1:numTrial
%             vel = get_vel(training_data, t_lag, tr, dir);
%             vels = cat(2, vels, vel);
%         end
%         velocities(dir).x = vels(1, :);
%         velocities(dir).y = vels(2, :);
%     end
%     
%     for dir = 1:numDir
%         for n = 1:length(selected_neurons)
%             fRates = [];
%             for tr = 1:numTrial
%                 spikeR = get_firingRate(training_data, selected_neurons(n), t_lag, tr, dir);
%                 fRates = cat(2, fRates, spikeR);
%             end
%             
%             firingRates(n, dir).frate = fRates;
%         end
%     end



