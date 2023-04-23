%%% Team Members: Coraline Beitone, Dorothy Cheng, Marco Cheng

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

      svm = SVM(X_train, y_train, @rbfKernel, 20, 0.01, 500);
      
      svmModels{numSvm} = svm;    
      
    end 

    modelParameters.svmModel = svmModels; % saved all 4 trained svm models 

    
%% Kalman filtering
    
% parameters definition
%     selected_neurons = 1:98; 
%     selected_neurons = [85,87,89,55,58,41,40,22,27,28,29,3,66,68];
%     selected_neurons = [4,14,18,29,34,36,48,55,67,68,75,77];
    selected_neurons = [3,7,23,27,28,29,40,41,55,58,61,66,67,68,85,87,88,89,96,98];
    
    t_lag = 20; % time window 20ms
    numState = 4; % number of states (x, y, v_x, v_y) - hand movements = system states

    
    %% function: get state function (x)
    function state = get_state(trainData, t_lag, nTrial, nDir, numState)
        length_spikes = size(trainData(nTrial, nDir).spikes, 2);
        numBin = floor((length_spikes - 320) / t_lag); 

        state = zeros(numState, numBin + 1); % for x,y

        % x, y, v_x, v_y
        bin = 320 + (0: t_lag: numBin*t_lag);
        state(1, :) = trainData(nTrial, nDir).handPos(1, bin); % x
        state(2, :) = trainData(nTrial, nDir).handPos(2, bin); % y
        state(3, 1:numBin) = diff(state(1, :)) / t_lag; % v_x
        state(4, 1:numBin) = diff(state(2, :)) / t_lag; % v_y
        
        state = state(:, 1:numBin); 
    end 


    %% function: get firing rate function (z)
    function fRate = get_firingRate(trainData, selected_neurons, t_lag, nTrial, nDir)
        length_spikes = size(trainData(nTrial, nDir).spikes, 2);
        numBin = floor((length_spikes - 320) / t_lag) - 1; 
        max_time = 320 + numBin * t_lag;
        
        numNeuron = length(selected_neurons);
        
        fRate = zeros(numNeuron, numBin);
        
        for n = 1:numNeuron
            spike = trainData(nTrial, nDir).spikes(selected_neurons(n), 321:max_time);
            fRate(n, :) = sum(spike(reshape(1:size(spike,2), t_lag, numBin)), 1) / t_lag; % calculate firing rate at each bin
        end
    end

    %% Get training data for Kalman filtering
    
    kalTrainDs = struct('state', cell(numTrial, numDir));
    start_states = cell(numDir, 1); 
    
    for dir = 1:numDir
        
        s0 = zeros(numState, numTrial); 
        
        for tr = 1:numTrial
            kalTrainDs(tr, dir).state = get_state(training_data, t_lag, tr, dir, numState);
            kalTrainDs(tr, dir).frate = get_firingRate(training_data, selected_neurons, t_lag, tr, dir);
            s0(:, tr) = kalTrainDs(tr, dir).state(:, 1); % get the first value of all states for each trial
        end
        
        start_states{dir} = s0;
    end

    %% Kalman filter training 
    
%   Function 1: Get states (hand movements) parameters (state model: A, W) 
    function [A, W] = get_states_params(states_ds) 
        numTrial = size(states_ds, 1);
        numState = size(states_ds(1).state, 1);

        A1 = zeros(numState);
        A2 = zeros(numState);
        W1 = zeros(numState);
        W2 = zeros(numState);
        
        max_numBins = zeros(numTrial, 1); % array saving all number of bins (length) of each trial
        for i = 1:numTrial
            max_numBins(i) = size(states_ds(i).state, 2);
        end
        
        for tr = 1:numTrial
            x1 = states_ds(tr).state(:, 2:max_numBins(tr));
            x2 = states_ds(tr).state(:, 1:(max_numBins(tr)-1));
            
            A1 = A1 + x1 * x2';
            A2 = A2 + x2 * x2';
            W1 = W1 + (1 / (max_numBins(tr)-1)) * (x1 * x1');
            W2 = W2 + (1 / (max_numBins(tr)-1)) * (x2 * x1');
        end
        
        A = A1 / A2;
        W = W1 - A * W2;
        
    end
    
%   Function 2: get observation parameters (observation model: H, Q)
    function [H, Q, Mz, Mx] = get_obs_params(states_ds)
        numTrial = size(states_ds, 1);
        numState = size(states_ds(1).state, 1);
        numNeuron = size(states_ds(1).frate, 1);
        
        H1 = zeros(numNeuron, numState);
        H2 = zeros(numState);
        Q1 = zeros(numNeuron, numNeuron);
        Q2 = zeros(numState, numNeuron);
        Mz = zeros(numNeuron, 1); % mean frate matrix
        Mx = zeros(numState, 1); % mean state matrix
        
        max_numBins = zeros(numTrial, 1); % array saving all number of bins (length) of each trial
        for i = 1:numTrial
            max_numBins(i) = size(states_ds(i).state, 2);
        end
        
        for tr = 1:numTrial
            mz = mean(states_ds(tr).frate, 2); % avg across all firing rates
            z = states_ds(tr).frate - mz; % wrt mean 
            
            mx = mean(states_ds(tr).state(:, 2:max_numBins(tr)), 2); % avg across all times (6 x 1)
            x = states_ds(tr).state(:, 2:max_numBins(tr)) - mx; % wrt mean
            
            Mz = Mz + mz;
            Mx = Mx + mx;
            
            H1 = H1 + z * x';
            H2 = H2 + x * x';
            Q1 = Q1 + (1 / max_numBins(tr)) * (z * z');
            Q2 = Q2 + (1 / max_numBins(tr)) * (x * z');

        end
        
        Mz = Mz / numTrial;
        Mx = Mx / numTrial;
        
        H = H1 / H2;
        Q = Q1 - H * Q2;

    end
    
    % Define 
    A = cell(numDir, 1);
    W = cell(numDir, 1);
    H = cell(numDir, 1);
    Q = cell(numDir, 1);
    
    Mz = cell(numDir, 1); % mean frate
    Mx = cell(numDir, 1); % mean state
    
    % get A, W, A, Q, Mz (mean z), Mx (mean x) for each direction (1-8)
    for dir = 1:numDir
        [A{dir}, W{dir}] = get_states_params(kalTrainDs(:, dir));
        [H{dir}, Q{dir}, Mz{dir}, Mx{dir}] = get_obs_params(kalTrainDs(:, dir));
    end
    
    % saving model parameters
    modelParameters.A = A;
    modelParameters.W = W;
    modelParameters.H = H;
    modelParameters.Q = Q;
    modelParameters.Mz = Mz;
    modelParameters.Mx = Mx;
    modelParameters.start_states = start_states;
    modelParameters.selectedNeurons = selected_neurons;
    modelParameters.state_num = numState;
    
end

