clear all;
close all;
clc;

%% Testing SVM accuracy

gtPred = decoding('monkeyKingdom');

accuracy = sum(gtPred(:, 1) == gtPred(:, 2))/size(gtPred, 1);
disp(['Accuracy of SVM direction prediction: ' num2str(accuracy)]);

