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
% - parameters: struct with classifier parameters, including:
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

function accuracy = test_gru(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, dispLogs)

 	% display logs only if multiprocessing is set
	if dispLogs
		disp('GRU validation test...')
	end
	gruLayerObj = gruLayer(parameters.numHiddenUnits,'OutputMode','last');

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
	parameters.inputSize = size(TRAIN_X{1},1);

	% definenetwork architecture (layer types and order)
	layers = [ ...
	sequenceInputLayer(parameters.inputSize)
	gruLayerObj
	fullyConnectedLayer(parameters.numClasses)
	softmaxLayer
	classificationLayer];

	% set training hyperparameters
	options = trainingOptions('adam', ...
	'ExecutionEnvironment',parameters.processingUnit, ...
	'GradientThreshold',parameters.gradientThreshold, ...
	'MaxEpochs',parameters.maxEpochs, ...
	'ValidationPatience', 5, ...
	'ValidationData', {TRAIN_X, TRAIN_Y}, ...
	'MiniBatchSize',parameters.miniBatchSize, ...
	'SequenceLength','longest', ...
	'Shuffle','never', ...
	'Verbose',0, ...
	'Plots','none',...
	'InitialLearnRate', parameters.initialLearnRate);

	% training
	net = trainNetwork(TRAIN_X, TRAIN_Y, layers, options); 
	% testing
	recognizedLabels = classify(net, TEST_X, 'MiniBatchSize', 1, 'SequenceLength', 'longest');
		
	% calculate classification accuracy
	recognizedSamplesCount = 0;
	for i = 1:length(TEST_Y)	
		if recognizedLabels(i) == TEST_Y(i)
			recognizedSamplesCount = recognizedSamplesCount + 1;
		end
	end
	accuracy = recognizedSamplesCount/length(TEST_Y);
end