%%% Team Members: WRITE YOUR TEAM MEMBERS' NAMES HERE

function [modelParameters] = positionEstimatorTraining_kalman(training_data)
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
      
      svmModels{numSvm} = svm;    
    end 

    modelParameters.svmModel = svmModels; % saved all 4 trained svm models 

    
%% Kalman filtering
    
    % Training data - state and activity extraction
    % state_model = x_k = [x, y, v_x, v_y, a_x, a_y]'
    % observation = z_k = [] (C x 1) firing rate over 20ms window at time k 
    
    % parameters definition
%     selected_neurons = 1:98; 
    selected_neurons = [85,87,89,55,58,41,40,22,27,28,29,3,66,68];
%     selected_neurons = [4,14,18,29,34,36,48,55,67,68,75,77];
    
    bin_size = 20; % time window 20ms
    order = 2; % get both velocity and acceleration
    
    % numTrials - number of trials (80)
    % numDir - number of directions (8) 
    
    
    %% function: get state function (x)
    function state = get_state(trainData, bin_size, nTrial, nDir, order)
        length_spikes = size(trainData(nTrial, nDir).spikes, 2);
        numBin = floor((length_spikes - 320) / bin_size) - order;  % why - order

        state_size = 2 * (order + 1);
        state = zeros(state_size, numBin + order + 1);

        % x and y
        binPoints = 320 + (0:bin_size:(numBin + order)*bin_size);
        state(1, :) = trainData(nTrial, nDir).handPos(1, binPoints);
        state(2, :) = trainData(nTrial, nDir).handPos(2, binPoints);

        % V_x, V_y and a_x, a_y
        for o = 1:order
            state(2*o +1, 1:(numBin + order)) = diff(state(2*(o-1)+1, :)) / bin_size;
            state(2*o +2, 1:(numBin + order)) = diff(state(2*(o-1)+2, :)) / bin_size;
        end

        state = state(:, 1:(numBin+1));
    end 

    %% function: get firing rate function (z)
    function fRate = get_firingRate(trainData, bin_size, nTrial, nDir, order, selected_neurons)
        length_spikes = size(trainData(nTrial, nDir).spikes, 2);
        numBin = floor((length_spikes - 320) / bin_size) - order;  % why - order
    
        max_time = 320 + numBin * bin_size;
        numNeuron = size(selected_neurons, 2);
        
        fRate = zeros(numNeuron, numBin);
        
        for n = 1:numNeuron
            neuron = selected_neurons(n);
            spike = trainData(nTrial, nDir).spikes(neuron, 321:max_time);
            idx = reshape(1:size(spike, 2), bin_size, numBin);
            spike_count = sum(spike(idx), 1);
            
            fRate(n, :) = spike_count / bin_size;
        end
    end

    %% Get training data for Kalman filtering
    kalTrainDs = struct('state', cell(numTrial, numDir));
    start_state = cell(numDir, 1);
    
    for dir = 1:numDir
        
        temp = zeros(2 * (order+1), numTrial);
        
        for tr = 1:numTrial
            kalTrainDs(tr, dir).state = get_state(training_data, bin_size, tr, dir, order);
            temp(:, tr) = kalTrainDs(tr, dir).state(:, 1);
            
            kalTrainDs(tr, dir).frate = get_firingRate(training_data, bin_size, tr, dir, order, selected_neurons);
        end
        
        start_state{dir} = temp;
    end

    %% train
    
%   Function 1: Get trajectory parameters (state model: A, W) 
%     function [A, W, Mx_p, Mx_f] = trajectory_params(states_ds) 
    function [A, W] = trajectory_params(states_ds) 
        state_size = size(states_ds(1).state, 1); % 6 states (x, y, vx, vy, ax, ay)
        numTrial = size(states_ds, 1);
        max_numBins = arrayfun(@(x)size(x.state, 2), states_ds); % array saving all number of bins (length) of each trial
        
        A1 = zeros(state_size);
        A2 = zeros(state_size);

        W1 = zeros(state_size);
        W2 = zeros(state_size);

        X1 = zeros(state_size, 1);
        X2 = zeros(state_size, 1);
        
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
    function [H, Q, Mz, Mx] = observation_params(states_ds)
        state_size = size(states_ds(1).state, 1);
        numNeuron = size(states_ds(1).frate, 1);
        numTrial = size(states_ds, 1);
        max_numBins = arrayfun(@(x)size(x.state, 2), states_ds);
        
        H1 = zeros(numNeuron, state_size);
        H2 = zeros(state_size);
        
        Q1 = zeros(numNeuron, numNeuron);
        Q2 = zeros(state_size, numNeuron);
        
        Mz = zeros(numNeuron, 1);
        Mx = zeros(state_size, 1);
        
        for tr = 1:numTrial
            mz = mean(states_ds(tr).frate, 2); % avg across all firing rates
            z = states_ds(tr).frate - repmat(mz, [1 max_numBins(tr)-1]);
            
            mx = mean(states_ds(tr).state(:, 2:max_numBins(tr)), 2); % avg across all times (6 x 1)
            x = states_ds(tr).state(:, 2:max_numBins(tr)) - repmat(mx, [1 max_numBins(tr)-1]);
            
            H1 = H1 + z * x';
            H2 = H2 + x * x';
            
            Q1 = Q1 + (1 / max_numBins(tr)) * (z * z');
            Q2 = Q2 + (1 / max_numBins(tr)) * (x * z');

            Mz = Mz + mz;
            Mx = Mx + mx;
        end
        
        H = H1 / H2;
        Q = Q1 - H * Q2;
        
        Mz = Mz / numTrial;
        Mx = Mx / numTrial;

    end
    
    % Define 
    A = cell(numDir, 1);
    W = cell(numDir, 1);
    H = cell(numDir, 1);
    Q = cell(numDir, 1);
    
    Mz = cell(numDir, 1);
    Mx = cell(numDir, 1);
    
    
    % get A, W, A, Q, Mz (mean z), Mx (mean x) for each direction (1-8)
    for dir = 1:numDir
        [A{dir}, W{dir}] = trajectory_params(kalTrainDs(:, dir));
        [H{dir}, Q{dir}, Mz{dir}, Mx{dir}] = observation_params(kalTrainDs(:, dir));
    end
    
    % saving model parameters
    modelParameters.A = A;
    modelParameters.W = W;
    modelParameters.H = H;
    modelParameters.Q = Q;
    modelParameters.Mz = Mz;
    modelParameters.Mx = Mx;
    modelParameters.start_state = start_state;
    modelParameters.selectedNeurons = selected_neurons;
    modelParameters.order = order;
    
end

