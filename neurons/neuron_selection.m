load('monkeydata_training.mat');

%% Raster plot of neurons over 100 trials + PSTH


numTrial = 100;
numDir = 1:2; % change the range for different direction (1:8)
numNeuron = 1:10; % change the range for different neurons (1:98)

time = 1:550;

smoothWidth = 30;
dt = 1;

figure(1);
for n = numNeuron
    for dir = numDir
        subplot(length(numNeuron), length(numDir), find(numNeuron == n)*length(numDir)-(length(numDir)-dir));
        counts_sum = zeros(1, length(time));
        for tr = 1:numTrial
%             time = 0:length(trial(tr, dir).spikes)-1;
            spike = trial(tr, dir).spikes(n, time);
            counts_sum = counts_sum + spike;
            scatter(time, tr*spike, '.', 'k', 'SizeData', 0.5);
            hold on;
        end
        line([320, 320], ylim, 'Color', 'k', 'LineWidth', 1);
        title(['Neuron ', num2str(n), ', Angle ', num2str(dir)]);
        hold on;
        
        % Average firing rate plot
        smooth_counts = smooth(counts_sum, smoothWidth);
        plot(time, smooth_counts * 1000 / (100 * dt), 'LineWidth', 2);
        
    end
end
