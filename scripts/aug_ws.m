%% Augmentation method:
%% WS - Window Slicing
%
% A. Le Guennec, S. Malinowski, R. Tavenard,
% Data augmentation for time series classification using convolutional neural networks,
% in: ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data,
% Riva Del Garda, Italy, 2016.
%
% This function performs augmentation by:
% 1. Randomly selecting a contiguous subsequence (window) within each time series,
% 2. Removing this subsequence to create a shorter but representative variant,
% 3. Generating multiple augmented samples per original sample by repeating this process,
% 4. Encouraging models to be robust to missing or incomplete temporal segments.
%
% Inputs:
% - train: cell array of multivariate time-series samples (features × time)
% - trainLabels: vector of class labels corresponding to training samples
% - nDraws: number of augmented samples to generate per original sample
%
% Outputs:
% - outTrain: cell array of augmented time-series samples with sliced windows removed
% - outTrainLabels: corresponding class labels for augmented data

function [outTrain, outTrainLabels] = aug_ws(train, trainLabels, nDraws)
    % Preallocate output cell arrays for efficiency
    outTrain = cell(1, length(trainLabels) * nDraws);
    outTrainLabels = zeros(1, length(trainLabels) * nDraws);
    xSize = size(train, 2);

    counter = 1;
    for i = 1:xSize
        % Current time series sample
        temp = train{i};
        nSamples = size(temp, 2);

        for iD = 1:nDraws
            % Define the size of the window to remove as random value up to one third of the sequence length
            a = 0;
            b = nSamples / 3;
            excerptLength = round((b - a) * rand() + a);

            % Randomly select start index of window to remove
            cutA = round(nSamples * rand());
            cutB = cutA + excerptLength;

            % Boundary checks for indices
            if cutB > nSamples
                cutB = nSamples;
            end
            if cutA == 0
                cutA = 1;
            end

            % Collect indices of the window to remove
            excerpt = cutA:cutB;

            % Remove the selected window (slice) from the time series
            tempAugmented = temp;
            tempAugmented(:, excerpt) = [];

            % Store augmented sample and corresponding label
            outTrain{counter} = tempAugmented;
            outTrainLabels(counter) = trainLabels(i);

            counter = counter + 1;
        end
    end
end
