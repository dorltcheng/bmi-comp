%%% Team Members: Coraline Beitone, Dorothy Cheng, Marco Cheng
%%% METHOD: SVMs + Kalman Filtering

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


%% Kalman filtering

    selected_neurons = [3,7,23,27,28,29,40,41,55,58,61,66,67,68,85,87,88,89,96,98]; % set of manually selected neurons 
    modelParameters.selected_neurons = selected_neurons;
    
    for dir = 1:numDir
        for tr = 1:numTrial
            [A,H,Q,W] = positionEstimatorTraining_one_trial(training_data(tr,dir),selected_neurons);
            Parameters.A{tr}=A;
            Parameters.H{tr}=H;
            Parameters.Q{tr}=Q;
            Parameters.W{tr}=W;
        end
    
        % average over trials to obtain a final set of parameters 
        % (A_(dir), H_(dir), Q_(dir), W_(dir)) for each direction
        modelParameters.A{dir}=sum(cat(3,Parameters.A{:}),3)./numTrial; 
        modelParameters.H{dir}=sum(cat(3,Parameters.H{:}),3)./numTrial; 
        modelParameters.W{dir}=sum(cat(3,Parameters.W{:}),3)./numTrial; 
        modelParameters.Q{dir}=sum(cat(3,Parameters.Q{:}),3)./numTrial; 

    end
    
    % Nested function for 'one trial' parameters estimation (A_(tr, dir), H_(tr, dir), Q_(tr, dir), W_(tr, dir)) for a given angle
    function [A,H,Q,W] = positionEstimatorTraining_one_trial(training_data,selected_neurons)

        % CONSTANTS
        lag = 20;% ms 
        nb_states = 4; % X Y Vx Vy 
        % NB: testing phase revealed that the inclusion of acceleration components in the state vector did not improved the performance of the decoder. 
        % Therefore, acceleration was excluded (only 4 states are used).
        Tstart = 320;% ms prediction starts 
        time_max = length(training_data.handPos); 
        time_max = time_max - Tstart; 

        % Build observation matrix z
        nb_bins = floor(time_max/lag); 
        for nr = 1:length(selected_neurons)
            neuron = selected_neurons(nr);
            spike = training_data.spikes(neuron, Tstart+1:(Tstart+nb_bins*lag));
            spike = reshape(spike, lag, nb_bins);
            spike_count = sum(spike, 1);
            z(nr, :) = spike_count / lag;
        end


        % Build state matrix x over time bins
        x = zeros(nb_states,nb_bins); % State Matrix 

        % Compute position every t=320+k*20ms 
        % x(1,:)=[X(320+20ms), X(320+40ms), ...., X(320+nb_bins*20ms)]
        % x(2,:)= [Y(320+20ms), Y(320+40ms), ...., Y(320+nb_bins*20ms)]

        for k = 1:nb_bins
            x(1,k) = training_data.handPos(1,Tstart+k*lag); % X
            x(2,k) = training_data.handPos(2,Tstart+k*lag); % Y
        end

        Pos_0 = training_data.handPos(1:2,Tstart); %(x0, y0) store initial position at t=320 ms

        % Compute velocity every t=320+k*20ms 
        % Vx(1,:)=(1/20)*[X(320+20ms)-X(320ms), X(320+40ms)-X(320+20ms), ...., X(320+nb_bins*20ms)-X(320+(nb_bins-1)*20ms)]
        % Vy(2,:)= (1/20)*[Y(320+20ms)-Y(320ms), Y(320+40ms)-Y(320+20ms), ...., Y(320+nb_bins*20ms)-Y(320+(nb_bins-1)*20ms)]
        for k = 1:nb_bins
            if k==1
                x(3,k) = (x(1,k)-Pos_0(1))/lag;
                x(4,k) = (x(2,k)-Pos_0(2))/lag;
            else
                x(3,k) = (x(1,k)-x(1,k-1))/lag;
                x(4,k) = (x(2,k)-x(2,k-1))/lag;
            end
        end

        % Parameters Estimations 
        % Reference: W. Wu, M. Black, Y. Gao, E. Bienenstock, M. Serruya, and J. Donoghue, "Inferring hand motion from multi-cell recordings in motor cortex 
        % using a kalman filter," (2002)

        % Useful Matrices  
        X1 = x(:,1:(end-1));
        X2 = x(:,2:end);

        % Compute A, H, W and Q
        A = X2*X1' / (X1*X1');
        W = X2*X2' - A*X1*X2';
        W = W / (nb_bins-1);
        H = z*x' / (x*x');
        Q = (z*z' - H*x*z');
        Q = Q / (nb_bins);

    end
    
end

