%% Augmentation method:
%% DBA - Dynamic Time Warping Barycenter Averaging
%
% H. I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, P.-A. Muller,
% "Data augmentation using synthetic data for time series classification
% with deep residual networks", CoRR abs/1808.02455 (2018).
%
% This function augments multivariate time-series data by:
% 1. Generating synthetic samples using Dynamic Time Warping Barycenter Averaging (DBA),
% 2. Selecting random subsets of samples from each class,
% 3. Iteratively aligning and averaging time-series sequences to create representative prototypes.
%
% Inputs:
% - train: cell array of multivariate time-series samples (dim × time)
% - trainLabels: corresponding class labels
% - nDraws: number of augmented versions to generate per class
%
% Outputs:
% - outTrain: augmented training data
% - outTrainLabels: corresponding labels for the augmented data

function [outTrain, outTrainLabels] = aug_dba(train, trainLabels, nDraws)
    numSamples = length(trainLabels) * nDraws;
    outTrain = cell(1, numSamples);
    outTrainLabels = zeros(1, numSamples);
    counter = 1;

    for iD = 1:nDraws
        for classLabel = min(trainLabels):max(trainLabels)
            % Extract samples of current class
            classSamples = train(trainLabels == classLabel);
            numClassSamples = numel(classSamples);

            if numClassSamples < 2
                continue; % Not enough data to apply DBA
            end

            % Randomly select a subset of samples (random number of samples from 1 to total available)
            numToSelect = randi([1, numClassSamples]);
            selected = classSamples(randperm(numClassSamples, numToSelect));

            if ~isempty(selected)
                outTrain{counter} = DBAmult(selected);
                outTrainLabels(counter) = classLabel;
                counter = counter + 1;
            end
        end
    end

    % Trim unused space (in case some iterations were skipped due to insufficient class data)
    outTrain(counter:end) = [];
    outTrainLabels(counter:end) = [];
end

% Perform full DBA averaging with multiple iterations
% Uses the medoid sequence as an initial prototype and iteratively refines it
function average = DBAmult(sequences)
    average = repmat(sequences{medoidIndex(sequences)}, 1); % Initialize with medoid
    for i = 1:15 % Fixed number of refinement iterations
        average = DBA_one_iterationMult(average, sequences);
    end
end

% Compute sum of squared DTW distances from sequence s to all others
function sos = sumOfSquares(s, sequences)
    sos = 0.0;
    for i = 1:numel(sequences)
        dist = dtw(s, sequences{i});
        sos = sos + dist^2;
    end
end

% Find the medoid (i.e., the most centrally located sequence in DTW space)
function index = medoidIndex(sequences)
    lowestInertia = Inf;
    index = -1;
    for i = 1:numel(sequences)
        inertia = sumOfSquares(sequences{i}, sequences);
        if inertia < lowestInertia
            lowestInertia = inertia;
            index = i;
        end
    end
end

% Single iteration of DBA: align all sequences to the current average and update
% This version supports multivariate sequences (dim × time)
function average = DBA_one_iterationMult(averageS, sequences)
    tupleAssociation = cell(size(averageS,1), size(averageS,2)); % Stores aligned values per (dim, time)
    costMatrix = zeros(1000,1000); % Preallocated DTW cost matrix
    pathMatrix = zeros(1000,1000); % Preallocated path backtracking

    for k = 1:numel(sequences)
        sequence = sequences{k};

        % Compute DTW distance matrix between current average and sequence
        costMatrix(1,1) = distanceTo(averageS(:,1), sequence(:,1));
        pathMatrix(1,1) = -1;
        for i = 2:size(averageS,2)
            costMatrix(i,1) = costMatrix(i-1,1) + distanceTo(averageS(:,i), sequence(:,1));
            pathMatrix(i,1) = 2;
        end
        for j = 2:size(sequence,2)
            costMatrix(1,j) = costMatrix(1,j-1) + distanceTo(sequence(:,j), averageS(:,1));
            pathMatrix(1,j) = 1;
        end
        for i = 2:size(averageS,2)
            for j = 2:size(sequence,2)
                [res, direction] = minWithIndex(...
                    costMatrix(i-1,j-1), ... % diagonal
                    costMatrix(i,j-1), ...   % left
                    costMatrix(i-1,j));      % up
                costMatrix(i,j) = res + distanceTo(averageS(:,i), sequence(:,j));
                pathMatrix(i,j) = direction;
            end
        end

        % Backtrack alignment path and collect aligned samples into tupleAssociation
        for d = 1:size(averageS,1) % For each dimension
            i = size(averageS,2);
            j = size(sequence,2);
            while true
                tupleAssociation{d,i}(end+1) = sequence(d,j); % Accumulate value at matched time point
                switch pathMatrix(i,j)
                    case 0, i = i - 1; j = j - 1;
                    case 1, j = j - 1;
                    case 2, i = i - 1;
                    otherwise, break;
                end
            end
        end
    end

    % Average over all collected alignments for each (dim, time) entry
    for d = 1:size(averageS,1)
        for t = 1:size(averageS,2)
            averageS(d,t) = mean(tupleAssociation{d,t});
        end
    end
    average = averageS;
end

% Compute squared Euclidean distance between vectors (used in DTW)
function dist = distanceTo(a, b)
    dist = sum((a - b).^2);
end

% Return minimum value and its index among three options (used in DTW)
function [val, idx] = minWithIndex(a, b, c)
    if a < b
        if a < c
            val = a; idx = 0;
        else
            val = c; idx = 2;
        end
    else
        if b < c
            val = b; idx = 1;
        else
            val = c; idx = 2;
        end
    end
end
