%% Single Validation Test Using Transformer Encoder
%
% This script performs a classification test using a custom Transformer-based
% neural network model for time series data. The model includes a projection
% layer, self-attention mechanism, feed-forward sub-block, and a classification head.
% The model is trained and evaluated on the given training and test sets.
%
% Inputs:
% - TRAIN_X: cell array of training sequences (features × time)
% - TRAIN_Y: vector of training labels
% - TEST_X: cell array of testing sequences
% - TEST_Y: vector of testing labels
% - parameters: structure with model and training hyperparameters:
%       .numHiddenUnits - internal model dimension (d_model)
%       .numHeads - number of attention heads
%       .dropout - dropout rate for regularization
%       .feedForwardSize - size of the FFN sub-block
%       .numClasses - number of output classes
%       .maxEpochs - number of training epochs
%       .miniBatchSize - size of training mini-batches
%       .initialLearnRate - learning rate for optimizer
%       .processingUnit - 'cpu' or 'gpu'
%       .gradientThreshold - gradient clipping threshold
% - dispLogs: boolean flag to display logs
%
% Output:
% - accuracy: classification accuracy on the test set

function accuracy = test_transformer(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, dispLogs)
    % display logs only if multiprocessing is set
    if dispLogs
        disp('Transformer validation test...')
    end

    %% label conversion and data preparation
    TRAIN_Y = categorical(TRAIN_Y);
    TEST_Y  = categorical(TEST_Y);

    % Sort sequences by length to improve training stability
    numObservations = numel(TRAIN_X);
    sequenceLengths = zeros(numObservations, 1);
    for i = 1:numObservations
         sequence = TRAIN_X{i};
         sequenceLengths(i) = size(sequence, 2);
    end
    [~, idx] = sort(sequenceLengths);
    TRAIN_X = TRAIN_X(idx);
    TRAIN_Y = TRAIN_Y(idx);
    
    % determine the input size from the training data
    parameters.inputSize = size(TRAIN_X{1}, 1);

    % ensure the internal dimension (numHiddenUnits) is divisible by numHeads
    if mod(parameters.numHiddenUnits, parameters.numHeads) ~= 0
         error('numHiddenUnits must be divisible by numHeads.');
    end
    numKeyChannels = parameters.numHiddenUnits / parameters.numHeads;

    %% build custom transformer encoder block within a layer graph

    % input layer: takes sequences with [features x timeSteps]
    inputLayer = sequenceInputLayer(parameters.inputSize, 'Name', 'input');
    
    % projection layer: maps input feature dimension to the internal dimension (d_model)
    projLayer = fullyConnectedLayer(parameters.numHiddenUnits, 'Name', 'proj');

    % --- self-attention sub-block ---
    selfAttn = selfAttentionLayer(parameters.numHeads, numKeyChannels, 'Name', 'self_attention');
    attnDrop = dropoutLayer(parameters.dropout, 'Name', 'attn_dropout');
    addition1 = additionLayer(2, 'Name', 'attn_add'); % residual connection
    norm1 = layerNormalizationLayer('Name', 'attn_norm');

    % --- feed-forward (FFN) sub-block ---
    fc1 = fullyConnectedLayer(parameters.feedForwardSize, 'Name', 'fc1'); 
    relu1 = reluLayer('Name', 'relu1');
    ffnDrop = dropoutLayer(parameters.dropout, 'Name', 'ffn_dropout');
    fc2 = fullyConnectedLayer(parameters.numHiddenUnits, 'Name', 'fc2');
    addition2 = additionLayer(2, 'Name', 'ffn_add'); % residual connection
    norm2 = layerNormalizationLayer('Name', 'ffn_norm');

    % --- classification head ---
    pool = globalAveragePooling1dLayer('Name','global_avg_pool');
    fc3 = fullyConnectedLayer(parameters.numClasses, 'Name','fc');
    softmx = softmaxLayer('Name','softmax');
    classOutput = classificationLayer('Name', 'classification');

    % assemble into a layer graph
    lgraph = layerGraph();
    lgraph = addLayers(lgraph, inputLayer);
    lgraph = addLayers(lgraph, projLayer);
    lgraph = addLayers(lgraph, selfAttn);
    lgraph = addLayers(lgraph, attnDrop);
    lgraph = addLayers(lgraph, addition1);
    lgraph = addLayers(lgraph, norm1);
    lgraph = addLayers(lgraph, fc1);
    lgraph = addLayers(lgraph, relu1);
    lgraph = addLayers(lgraph, ffnDrop);
    lgraph = addLayers(lgraph, fc2);
    lgraph = addLayers(lgraph, addition2);
    lgraph = addLayers(lgraph, norm2);
    lgraph = addLayers(lgraph, pool);
    lgraph = addLayers(lgraph, fc3);
    lgraph = addLayers(lgraph, softmx);
    lgraph = addLayers(lgraph, classOutput);
    
    % --- connect layers ---
    lgraph = connectLayers(lgraph, 'input', 'proj');
    lgraph = connectLayers(lgraph, 'proj', 'self_attention');
    lgraph = connectLayers(lgraph, 'self_attention', 'attn_dropout');
    lgraph = connectLayers(lgraph, 'proj', 'attn_add/in2'); % skip connection
    lgraph = connectLayers(lgraph, 'attn_dropout', 'attn_add/in1');
    lgraph = connectLayers(lgraph, 'attn_add', 'attn_norm');
    lgraph = connectLayers(lgraph, 'attn_norm', 'fc1');
    lgraph = connectLayers(lgraph, 'fc1', 'relu1');
    lgraph = connectLayers(lgraph, 'relu1', 'ffn_dropout');
    lgraph = connectLayers(lgraph, 'ffn_dropout', 'fc2');
    lgraph = connectLayers(lgraph, 'attn_norm', 'ffn_add/in2'); % skip connection
    lgraph = connectLayers(lgraph, 'fc2', 'ffn_add/in1');
    lgraph = connectLayers(lgraph, 'ffn_add', 'ffn_norm');
    lgraph = connectLayers(lgraph, 'ffn_norm', 'global_avg_pool');
    lgraph = connectLayers(lgraph, 'global_avg_pool', 'fc');
    lgraph = connectLayers(lgraph, 'fc', 'softmax');
    lgraph = connectLayers(lgraph, 'softmax', 'classification');
    
    %% set training options
    options = trainingOptions('adam', ...
        'ExecutionEnvironment', parameters.processingUnit, ...
        'GradientThreshold', parameters.gradientThreshold, ...
        'MaxEpochs', parameters.maxEpochs, ...
        'ValidationPatience', 5, ...
        'ValidationData', {TRAIN_X, TRAIN_Y}, ...
        'MiniBatchSize', parameters.miniBatchSize, ...
        'SequenceLength', 'longest', ...
        'Shuffle', 'never', ...
        'Verbose', 0, ...
        'Plots', 'none', ...
        'InitialLearnRate', parameters.initialLearnRate);

    %% train the network
    net = trainNetwork(TRAIN_X, TRAIN_Y, lgraph, options);
    
    %% testing
    recognizedLabels = classify(net, TEST_X, 'MiniBatchSize', 1, 'SequenceLength', 'longest');
    recognizedSamplesCount = sum(recognizedLabels' == TEST_Y);
    accuracy = recognizedSamplesCount / numel(TEST_Y);
end
