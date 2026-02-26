%% Single Validation Test Using LSTM Classifier
%
% This script performs a classification test using an LSTM or BiLSTM
% neural network model on time series data. It includes training the
% network and evaluating its accuracy on the test set.
%
% Inputs:
% - TRAIN_X: cell array of training sequences (features × time)
% - TRAIN_Y: vector of training labels
% - TEST_X: cell array of testing sequences
% - TEST_Y: vector of testing labels
% - params: structure with model and training hyperparameters:
%       .bidirectional - use BiLSTM if true, else LSTM
%       .numHiddenUnits - number of hidden units in LSTM layer
%       .numClasses - number of output classes
%       .maxEpochs - number of training epochs
%       .miniBatchSize - size of training mini-batches
%       .initialLearnRate - learning rate for optimizer
%       .processingUnit - e.g., 'cpu' or 'gpu'
%       .gradientThreshold - gradient clipping threshold
% - dispLogs: boolean flag to display logs
%
% Output:
% - accuracy: classification accuracy on the test set
% - metrics: table with other metrics

function [accuracy, metrics] = test_lstm(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, params, dispLogs)

    %% Choose LSTM or BiLSTM layer
 	% display logs only if multiprocessing is set
	if params.bidirectional
		if dispLogs
			disp('BiLSTM validation test...')
		end
		lstmLayerObj = lstmLayer(params.numHiddenUnits,'OutputMode','last');
	else
		if dispLogs
			disp('LSTM validation test...')
		end
		lstmLayerObj = bilstmLayer(params.numHiddenUnits,'OutputMode','last');
	end

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
	lstmLayerObj
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