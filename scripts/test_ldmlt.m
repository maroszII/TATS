%% Single Validation Test Using LDMLT Classifier
%
% This function evaluates the classification accuracy of the LDMLT (Local Discriminant
% Metric Learning for Time Series) classifier. It performs metric learning on the training 
% data and evaluates the resulting model using k-NN classification on the test set.
%
% Inputs:
% - TRAIN_X: cell array of training sequences (features × time)
% - TRAIN_Y: vector of training labels
% - TEST_X: cell array of test sequences (features × time)
% - TEST_Y: vector of test labels
% - parameters: struct with the following fields:
%       .epochs - number of training cycles (mapped to `cycle` in LDMLT)
%       .k - number of nearest neighbors for classification
%       .[other fields specific to LDMLT, e.g., sigma, lambda, etc.]
% - dispLogs: boolean flag to display logs
%
% Output:
% - accuracy: classification accuracy on the test set

function accuracy = test_ldmlt(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, dispLogs)

    %% Add LDMLT library path
    addpath('scripts/LDMLT');

    %% Optional logging
    if dispLogs
        disp('Starting LDMLT validation test...');
    end

    %% Transpose data to expected format (samples × features)
    % LDMLT expects time series as (time × features), so we transpose each cell
    TRAIN_X = cellfun(@(x) x.', TRAIN_X, 'UniformOutput', false);
    TEST_X = cellfun(@(x) x.', TEST_X, 'UniformOutput', false);

    %% Set training cycles in LDMLT parameters
    parameters.cycle = parameters.epochs;

    %% Train LDMLT metric learning model
    M = LDMLT_TS(TRAIN_X, TRAIN_Y, parameters);

    %% Classify test samples using k-NN in learned metric space
    recognizedLabels = KNN_TS(TRAIN_X, TRAIN_Y, TEST_X, M, parameters.k);

    %% Evaluate classification accuracy
    % The output of KNN_TS is a matrix of size (k × numTestSamples).
    % We compare the top-1 prediction (row `k`) with ground truth labels.
    correctCount = 0;
    for i = 1:length(TEST_Y)
        if recognizedLabels(parameters.k, i) == TEST_Y(i)
            correctCount = correctCount + 1;
        end
    end
    accuracy = correctCount / length(TEST_Y);
end
