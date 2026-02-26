% normalizeMinMax performs time series data normalization by scaling it to the [0, 1] range.
%
% Inputs:
%   - trainingSet: Cell array containing training time series (Features x Time)
%   - testingSet: Cell array containing testing time series (Features x Time)
%
% Outputs:
%   - normalizedTrain: normalized training set
%   - normalizedTest: normalized testing set

function [normalizedTrain, normalizedTest] = normalizeMinMax(trainingSet, testingSet)
    % Concatenate all training data to find global min/max per feature
    allTrainData = cat(2, trainingSet{:});
    
    % Calculate Min and Max for each feature (row)
    minVal = min(allTrainData, [], 2);
    maxVal = max(allTrainData, [], 2);
    
    % Calculate the range.
    rangeVal = maxVal - minVal;
    
    % Handle constant features where range is 0 to avoid division by zero
    rangeVal(rangeVal == 0) = 1; 
    
    % Apply normalization to the Training Set
    normalizedTrain = trainingSet;
    numTrain = numel(trainingSet);
    
    for i = 1:numTrain
        % (X - Min) / (Max - Min)
        normalizedTrain{i} = (trainingSet{i} - minVal) ./ rangeVal;
    end
    
    % Apply the same normalization (using training stats) to the Testing Set
    normalizedTest = testingSet;
    numTest = numel(testingSet);
    
    for i = 1:numTest
        normalizedTest{i} = (testingSet{i} - minVal) ./ rangeVal;
    end
end