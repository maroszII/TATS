%% k-NN with Dynamic Time Warping (DTW) Validation Test
%
% This function performs classification using a k-Nearest Neighbors classifier
% where the distance metric is Dynamic Time Warping (DTW). For each test sample,
% distances to all training samples are computed, and the class is assigned
% based on the majority vote among the k closest neighbors.
%
% Inputs:
% - TRAIN_X: cell array of training samples (features × time)
% - TRAIN_Y: vector of training labels
% - TEST_X: cell array of testing samples
% - TEST_Y: vector of testing labels
% - params: struct with classifier parameters, including:
%       .windowSize - Sakoe-Chiba band width for DTW
%       .metric - metric used in DTW (e.g., 'euclidean')
%       .k - number of neighbors for k-NN
% - dispLogs: boolean flag to display progress messages
%
% Output:
% - accuracy: classification accuracy on the test set
% - metrics: table with other metrics

function [accuracy, metrics] = test_dtw(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, params, dispLogs)

    %% Optional log display
    if dispLogs
        disp('Starting k-NN with DTW validation test...');
    end

    recognizedLabels = zeros(length(TEST_X), 1);
    recognizedSamplesCount = 0;
    
    %% Classification loop for each test sample
    for i = 1:length(TEST_X)
        distances = zeros(length(TRAIN_X), 1);
        labels = zeros(length(TRAIN_X), 1);
        
        % Calculate DTW distance from test sample i to all training samples
        for j = 1:length(TRAIN_X)
            distances(j) = dtw(TEST_X{i}, TRAIN_X{j}, params.windowSize, params.metric);
            labels(j) = TRAIN_Y(j);
        end
        
        % Find indices of k smallest distances (nearest neighbors)
        [~, sortedIndices] = sort(distances);
        kNearestLabels = labels(sortedIndices(1:params.k));
        
        % Assign label based on majority vote among k neighbors
        recognizedLabel = mode(kNearestLabels);
        recognizedLabels(i) = recognizedLabel;
    end

    %% Calculate overall accuracy
    [accuracy, metrics] = calculate_metrics(categorical(TEST_Y), categorical(recognizedLabels));
end
