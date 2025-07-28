% % Augmentation method:
% % Jittering
%
% T. T. Um, F. M. J. Pfister, D. Pichler, S. Endo, M. Lang, S. Hirche,
% U. Fietzek, D. Kulić,
% "Data augmentation of wearable sensor data for Parkinson’s disease monitoring using convolutional neural networks",
% ACM International Conference on Multimodal Interaction, ACM, 2017.
%
% This function augments multivariate time-series data by adding Gaussian noise (jittering) to simulate sensor variability.
%
% Inputs:
% - train: cell array of multivariate time-series samples (dim × time)
% - trainLabels: corresponding class labels
% - nDraws: number of augmented versions to generate per sample
% - stdev: standard deviation of Gaussian noise (default = 2.0)
%
% Outputs:
% - outTrain: augmented training data with added noise
% - outTrainLabels: corresponding labels for the augmented data

function [outTrain, outTrainLabels] = aug_jitter(train, trainLabels, nDraws, stdev)
    if nargin < 4
        stdev = 2.0; % default noise standard deviation
    end

    outTrain = cell(1, length(train) * nDraws);
    outTrainLabels = zeros(1, length(train) * nDraws);
    xSize = length(train);

    counter = 1;
    for i = 1:xSize
        temp = train{i};
        for iD = 1:nDraws
            % Generate Gaussian noise matching the time-series size
            noise = stdev * randn(size(temp));

            % Add noise to the original time-series (jittering)
            noisy_series = temp + noise;

            % Store augmented sample and corresponding label
            outTrain{counter} = noisy_series;
            outTrainLabels(counter) = trainLabels(i);

            counter = counter + 1;
        end
    end
end
