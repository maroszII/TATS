% normalizeZScore performs time series z-score data normalization by normalization.
%
% Inputs:
%   - trainingSet: Cell array containing training time series (Features x Time)
%   - testingSet: Cell array containing testing time series (Features x Time)
%
% Outputs:
%   - normalizedTrain: normalized training set
%   - normalizedTest: normalized testing set
	
function [normalizedTrain, normalizedTest] = normalizeZScore(trainingSet, testingSet)   
    % Concatenate all training data to calculate global statistics per feature
    allTrainData = cat(2, trainingSet{:});
    
    % Calculate Mean and Standard Deviation for each feature (row)
    mu = mean(allTrainData, 2);
    sigma = std(allTrainData, 0, 2);
    
    % Handle constant features where sigma is 0 to avoid division by zero (NaNs)
    % We replace 0 with 1, so the value becomes (x - mu) / 1 = 0 (centered)
    sigma(sigma == 0) = 1;
    
    % Apply normalization to the Training Set
    normalizedTrain = trainingSet; % Initialize with original structure
    numTrain = numel(trainingSet);
    
    for i = 1:numTrain
        % (X - Mean) / Std
        % MATLAB handles dimension broadcasting automatically here
        normalizedTrain{i} = (trainingSet{i} - mu) ./ sigma;
    end
    
    % Apply the SAME normalization (using training stats) to the Testing Set
    normalizedTest = testingSet; % Initialize with original structure
    numTest = numel(testingSet);
    
    for i = 1:numTest
        normalizedTest{i} = (testingSet{i} - mu) ./ sigma;
    end
end