function accuracy = test(trainingData, testingData, augFunction, targetAugSamNum, parameters, multiprocessing)

for i = 1:length(trainingData)
	TRAIN_X{i} = trainingData{i,1}';
	TRAIN_Y(i) = str2num(trainingData{i,2});
end

for i = 1:length(testingData)
	TEST_X{i} = testingData{i,1}';
	TEST_Y(i) = str2num(testingData{i,2});
end

% Augmenting: generate at least targetAugSamNum samples
if isa(augFunction, 'function_handle')
    if ~multiprocessing
        disp('Augmenting...')
    end
    outTrain = [];
    outTrainLabels = [];
    while true
        if strcmp(func2str(augFunction), 'aug_adder')
            [out_temp, out_lab_temp] = augment(TRAIN_X,TRAIN_Y,augFunction,1); 
            TEST_X = change_representation(TEST_X);
        else
            [out_temp, out_lab_temp] = augment(TRAIN_X,TRAIN_Y,augFunction,1);
        end
    
	    outTrain = [outTrain out_temp];
	    outTrainLabels = [outTrainLabels out_lab_temp];
	    if length(outTrain) >= targetAugSamNum
		    break
	    end
    end
    
    % Limit the number of samples to be exactly targetAugSamNum
    while length(outTrain) > targetAugSamNum
	    index = randi(length(outTrain));
	    outTrain(index) = [];
	    outTrainLabels(index) = [];
    end
    
    % Add augmented samples to original samples
    TRAIN_X = [TRAIN_X outTrain];
    TRAIN_Y = [TRAIN_Y outTrainLabels];
end
 

if strcmpi(parameters.classifier,'DTW')
    accuracy = test_dtw(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, multiprocessing);
else
    accuracy = test_lstm(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, multiprocessing);
end