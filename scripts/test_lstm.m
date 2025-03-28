function accuracy = test_lstm(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, multiprocessing)

if parameters.bidirectional
    if ~multiprocessing
        disp('BiLSTM validation test...')
    end
    lstmLayerObj = bilstmLayer(parameters.numHiddenUnits,'OutputMode','last');
else
    if ~multiprocessing
        disp('LSTM validation test...')
    end
    lstmLayerObj = lstmLayer(parameters.numHiddenUnits,'OutputMode','last');
end

TRAIN_Y = categorical(TRAIN_Y);
TEST_Y = categorical(TEST_Y);

% Prepare training data for padding
numObservations = numel(TRAIN_X);
sequenceLengths = zeros(numObservations, 1);
for i=1:numObservations
	sequence = TRAIN_X{i};
	sequenceLengths(i) = size(sequence,2);
end

[~,idx] = sort(sequenceLengths);
TRAIN_X = TRAIN_X(idx);
TRAIN_Y = TRAIN_Y(idx);

% Prepare training data for padding
parameters.inputSize = size(TRAIN_X{1},1);

layers = [ ...
sequenceInputLayer(parameters.inputSize)
lstmLayerObj
fullyConnectedLayer(parameters.numClasses)
softmaxLayer
classificationLayer];

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
net = trainNetwork(TRAIN_X, TRAIN_Y, layers, options); % Training 
recognizedLabels = classify(net, TEST_X, 'MiniBatchSize', 1, 'SequenceLength', 'longest'); % Classification 
recognizedSamplesCount = 0;
for i = 1:length(TEST_Y)	
	if recognizedLabels(i) == TEST_Y(i)
		recognizedSamplesCount = recognizedSamplesCount + 1;
	end
end 
accuracy = recognizedSamplesCount/length(TEST_Y);