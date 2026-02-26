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
% - params: struct with the following fields:
%       .epochs - number of training cycles (mapped to `cycle` in LDMLT)
%       .k - number of nearest neighbors for classification
%       .[other fields specific to LDMLT, e.g., sigma, lambda, etc.]
% - dispLogs: boolean flag to display logs
%
% Output:
% - accuracy: classification accuracy on the test set
% - metrics: table with other metrics

function [accuracy, metrics] = test_ldmlt(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, params, dispLogs)

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
    params.cycle = params.epochs;

    %% Train LDMLT metric learning model
    M = LDMLT_TS(TRAIN_X, TRAIN_Y, params);

    %% Classify test samples using k-NN in learned metric space
    recognizedLabels = KNN_TS(TRAIN_X, TRAIN_Y, TEST_X, M, params.k);

    %% Calculate classification accuracy
    [accuracy, metrics] = calculate_metrics(categorical(TEST_Y), categorical(recognizedLabels));
end
