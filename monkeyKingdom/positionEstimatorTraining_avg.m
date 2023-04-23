%%% Team Members: Coraline Beitone, Dorothy Cheng, Marco Cheng

function [modelParameters] = positionEstimatorTraining_avg(training_data)

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


    %% Averaging

    % Function to calculate average trajectory across all trials
    function [X_avg,Y_avg] = compute_mean(X,Y,l)
        l_s = unique(sort(l));
        N = size(X,1);
        for i = 1:length(l_s)
            if i == 1
                X_avg(1:l_s(i)) = sum(X(:,1:l_s(i)),1)/N;
                Y_avg(1:l_s(i)) = sum(Y(:,1:l_s(i)),1)/N;
            else
                tmp_X = X(:,(l_s(i-1)+1):l_s(i));
                tmp_Y = Y(:,(l_s(i-1)+1):l_s(i));
                X_avg(l_s(i-1)+1:l_s(i)) = sum(tmp_X,1)/N;
                Y_avg(l_s(i-1)+1:l_s(i)) = sum(tmp_Y,1)/N;
            end
            N = N - length(find(l==l_s(i)));

        end
    end

    max_val = 1000;

    for a=1:numDir
        X = zeros(numTrial,max_val);
        Y = zeros(numTrial,max_val);
        
        l=[];
        for t = 1:numTrial
            x = training_data(t,a).handPos(1,t_length:end);
            y = training_data(t,a).handPos(2,t_length:end);
            l = [l,length(x)];
            X(t,1:l(end)) = x;
            Y(t,1:l(end)) = y;
        end

        [X_avg,Y_avg] = compute_mean(X,Y,l);
        modelParameters.X{a} = X_avg;
        modelParameters.Y{a} = Y_avg;

    end

    

end
