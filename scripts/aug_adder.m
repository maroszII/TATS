%% Augmentation method: Adder
% 
% Based on:
% 
% *M. Oszust, D. Warchoł, Time series augmentation with time-scale
% modifications and piecewise aggregate approximation for human action 
% recognition, in: 2022 IEEE 34th International Conference on Tools 
% with Artificial Intelligence (ICTAI), 2022, pp. 700–704.*
%
% This function augments multivariate time-series data by:
% 1. Resampling the series to a new length via interpolation (time scaling),
% 2. Applying Piecewise Aggregate Approximation (PAA): splitting the signal into
%    equal-length segments and computing the mean of each segment.
%
% Inputs:
% - train: cell array of multivariate time-series samples (dim × time)
% - trainLabels: corresponding class labels
% - nDraws: number of augmented versions to generate per sample
% - segNum: number of segments for PAA (default = 10)
%
% Outputs:
% - outTrain: augmented training data
% - outTrainLabels: corresponding labels for the augmented data

function [outTrain,outTrainLabels] = aug_adder(train,trainLabels,nDraws,segNum)

% Set default number of segments if not specified
if nargin < 4
    segNum = 10;
end

% Preallocate output arrays
outTrain = cell(1, length(trainLabels) * nDraws); 
outTrainLabels = [];
for i = 1:nDraws
    outTrainLabels = [outTrainLabels trainLabels];
end

segNumOrig = segNum;

% Determine minimum and maximum series lengths in the training set
maxDl = 0;
minDl = 10e10;
for i = 1:length(train)
    mtemp = length(train{i});
    maxDl = max(maxDl, mtemp);
    minDl = min(minDl, mtemp);
end	

% Define allowed interpolation lengths (multiples of segment count)
maxDlM = ceil(maxDl * 2 / segNum) * segNum;
minDlM = segNum;
a = minDlM:segNum:maxDlM;

counter = 1;
for lP = 1:nDraws
    for i = 1:length(train)
        tempI = train{i};

        % Skip short samples (less than 3 time steps)
        if size(tempI, 2) < 3
            outTrain{counter} = tempI;
            outTrainLabels(counter) = trainLabels(i);
            counter = counter + 1;
            continue
        end

        % Detemine number of segments based on random value from set of input numbers        
        segNum = ceil(segNumOrig * 0.8):ceil(segNumOrig * 1.2);
        segNum = segNum(randperm(length(segNum))); 
        segNum = segNum(1);
        maxDlM = ceil(maxDl * 2 / segNum) * segNum;
        minDlM = segNum;
        a = minDlM:segNum:maxDlM;
         

        % Randomly choose new length for resampling
        data_len = size(tempI, 2);
        a_rand = a(randperm(length(a)));
        aa = a_rand(1);
        data_len = ceil(data_len / aa) * aa;

        % === Step 1: Interpolation (time-scale modification) ===
        % Resample the time-series to the new length
        data = tempI';
        newLen = data_len;
        [initSize1, initSize2] = ndgrid(1:size(data, 1), 1:size(data, 2));
        [newSize1, newSize2] = ndgrid(linspace(1, size(data, 1), newLen), 1:size(data, 2));
        newData = interpn(initSize1, initSize2, data, newSize1, newSize2);

        % === Step 2: Piecewise Aggregate Approximation (PAA) ===
        % Divide each dimension into 'segNum' segments and average each
        segSiz = floor(data_len / segNum);
        tempR = [];

        for d = 1:size(newData, 2)
            data = newData(:, d);		
            data = (data - mean(data)) / std(data);  % Z-normalization

            % Reshape into segments (segSiz × segNum) and compute segment means
            dane = reshape(data, segSiz, segNum); 
            dane(isnan(dane)) = 0;  % Handle numerical issues
            segments = mean(dane);  % One value per segment (PAA)

            % Ensure no NaNs remain
            segments(isnan(segments)) = 0;

            % Concatenate features from each dimension
            tempR = [tempR, segments'];
        end

        % Store the final augmented sample
        outTrain{counter} = tempR';  % Shape: [feature × segment]
        counter = counter + 1;
    end
end