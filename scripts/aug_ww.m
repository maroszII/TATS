%% Augmentation method:
%% WW - Window Warping
%
% A. Le Guennec, S. Malinowski, R. Tavenard,
% Data augmentation for time series classification using convolutional neural networks,
% in: ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data,
% Riva Del Garda, Italy, 2016.
%
% This function performs augmentation by:
% 1. Selecting a random contiguous subsequence ("window") from each time series,
% 2. Warping this window non-linearly by random stretching (duplicating indices),
% 3. Re-inserting the warped window back into the time series.
%
% Inputs:
% - train: cell array of multivariate time-series samples (features × time)
% - trainLabels: vector of class labels for the training samples
% - nDraws: number of augmented samples to generate per original sample
%
% Outputs:
% - outTrain: cell array of augmented time-series data with warped windows
% - outTrainLabels: corresponding labels for augmented data

function [outTrain, outTrainLabels] = aug_ww(train, trainLabels, nDraws)
    % Preallocate output arrays for performance
    outTrain = cell(1, length(trainLabels) * nDraws);
    outTrainLabels = zeros(1, length(trainLabels) * nDraws);
    xSize = size(train, 2);

    counter = 1;
    for i = 1:xSize
        temp = train{i};
        nSamples = size(temp, 2);

        for iD = 1:nDraws
            % Define max window size as up to one third of the sequence length
            a = 0;
            b = nSamples / 3;
            excerptLength = round((b - a) * rand() + a);

            % Random start and end indices for the window
            cutA = round(nSamples * rand());
            cutB = cutA + excerptLength;

            % Boundary checks
            if cutB > nSamples
                cutB = nSamples;
            end
            if cutA == 0
                cutA = 1;
            end

            % Collect indices of the window to warp
            excerpt = cutA:cutB;          
            excerpt2 = [];

            % Non-linear stretching: randomly duplicate indices to simulate temporal dilation
            for j = excerpt
                % Randomly repeat index once or twice (uniformly)
                for los = 1:round((2 - 1) * rand() + 1)
                    excerpt2 = [excerpt2, j];
                end
            end

            % Append original indices to preserve global structure
            for los = 1:nSamples
                excerpt2 = [excerpt2, los];
            end

            % Combine and sort indices to form new warped timeline
            excerpt = [excerpt2, excerpt];
            excerpt = sort(excerpt);

            % Apply warping by reindexing the original series with warped indices
            tempWarped = temp(:, excerpt);
    
            outTrain{counter} = tempWarped;
            outTrainLabels(counter) = trainLabels(i);

            counter = counter + 1;
        end
    end
end
