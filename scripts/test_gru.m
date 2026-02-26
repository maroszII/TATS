%% Single Validation Test Using Gated Recurrent Unit (GRU) Classifier
%
% This function performs classification of time-series data using a GRU-based
% recurrent neural network. It trains the GRU on the training data and evaluates
% accuracy on the test set.
%
% Inputs:
% - TRAIN_X: cell array of training sequences (features × time)
% - TRAIN_Y: vector of training labels
% - TEST_X: cell array of test sequences
% - TEST_Y: vector of test labels
% - params: struct with classifier parameters, including:
%       .numHiddenUnits - number of hidden units in the GRU layer
%       .numClasses - number of output classes
%       .processingUnit - 'cpu' or 'gpu' for training environment
%       .gradientThreshold - gradient clipping threshold
%       .maxEpochs - maximum number of training epochs
%       .miniBatchSize - size of mini-batches for training
%       .initialLearnRate - initial learning rate for optimizer
% - dispLogs: boolean flag to display training logs
%
% Output:
% - accuracy: classification accuracy on the test set
% - metrics: table with other metrics

function [accuracy, metrics] = test_gru(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, params, dispLogs)

 	% display logs only if multiprocessing is set
	if dispLogs
		disp('GRU validation test...')
	end
	gruLayerObj = gruLayer(params.numHiddenUnits,'OutputMode','last');

	% get class labels
	TRAIN_Y = categorical(TRAIN_Y);
	TEST_Y = categorical(TEST_Y);

	% prepare training data for padding
	numObservations = numel(TRAIN_X);
	sequenceLengths = zeros(numObservations, 1);
	for i=1:numObservations
		sequence = TRAIN_X{i};
		sequenceLengths(i) = size(sequence,2);
	end

	% sort sequence lengths
	[~,idx] = sort(sequenceLengths);
	TRAIN_X = TRAIN_X(idx);
	TRAIN_Y = TRAIN_Y(idx);

	% prepare training data for padding
	params.inputSize = size(TRAIN_X{1},1);

	% definenetwork architecture (layer types and order)
	layers = [ ...
	sequenceInputLayer(params.inputSize)
	gruLayerObj
	fullyConnectedLayer(params.numClasses)
	softmaxLayer
	classificationLayer];

	% set training hyperparameters
	options = trainingOptions('adam', ...
	'ExecutionEnvironment',params.processingUnit, ...
	'GradientThreshold',params.gradientThreshold, ...
	'MaxEpochs',params.maxEpochs, ...
	'ValidationPatience', 5, ...
	'ValidationData', {TRAIN_X, TRAIN_Y}, ...
	'MiniBatchSize',params.miniBatchSize, ...
	'SequenceLength','longest', ...
	'Shuffle','never', ...
	'Verbose',0, ...
	'Plots','none',...
	'InitialLearnRate', params.initialLearnRate);

	% training
	net = trainNetwork(TRAIN_X, TRAIN_Y, layers, options); 
	% testing
	recognizedLabels = classify(net, TEST_X, 'MiniBatchSize', 1, 'SequenceLength', 'longest');
		
	% calculate classification accuracy
	[accuracy, metrics] = calculate_metrics(TEST_Y, recognizedLabels);
end