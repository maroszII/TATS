%% Single validation test with optional data augmentation
%
% This function trains a classifier on training data (possibly augmented),
% then evaluates classification accuracy on a testing set.
%
% Inputs:
% - trainingData: cell array of training samples and labels
% - testingData: cell array of testing samples and labels
% - augFunction: handle to augmentation function or 'no aug' string
% - targetAugSamNum: desired number of augmented samples to generate
% - params: structure containing general parameters and classifier-specific hyperparameters
% - multiprocessing: boolean flag indicating if multiprocessing is enabled
%
% Output:
% - accuracy: classification accuracy on the test set
% - metrics: table with other metrics

function [accuracy, metrics, augTime] = test(trainingData, testingData, augFunction, targetAugSamNum, params, multiprocessing)
    %% Prepare training and testing data matrices and label vectors
    % Convert cell data to format: TRAIN_X{i} = features (channels × time), TRAIN_Y(i) = label number
    for i = 1:length(trainingData)
        TRAIN_X{i} = trainingData{i,1}'; % transpose: time × features -> features × time
        TRAIN_Y(i) = str2double(trainingData{i,2}); % convert label string to number
    end

    for i = 1:length(testingData)
        TEST_X{i} = testingData{i,1}';
        TEST_Y(i) = str2double(testingData{i,2});
    end
	
	% Before augmentation sets may be normalized (if specified by the parameter)
	if params.normalization == 1
		[TRAIN_X, TEST_X] = normalizeMinMax(TRAIN_X, TEST_X);
	elseif params.normalization == 2
		[TRAIN_X, TEST_X] = normalizeZScore(TRAIN_X, TEST_X);
	end

    %% Data augmentation (if augmentation function provided)
    % Generate augmented data until target sample count is reached
    if isa(augFunction, 'function_handle')
        if ~multiprocessing
            disp('Augmenting training data...');
        end
        
        outTrain = [];
        outTrainLabels = [];
        
        tStart = tic;
        % Repeat augmentation calls until enough augmented samples generated
        while true
            [out_temp, out_lab_temp] = augment(TRAIN_X, TRAIN_Y, augFunction, 1); 
                        
            outTrain = [outTrain out_temp];
            outTrainLabels = [outTrainLabels out_lab_temp];
            
            if length(outTrain) >= targetAugSamNum
                break; % Stop augmenting once sufficient samples generated
            end
        end

        % Special handling: aug_adder requires change in testing set representation
        if strcmp(func2str(augFunction), 'aug_adder')
            TEST_X =  change_representation(TEST_X); % Update testing set accordingly
			TRAIN_X =  change_representation(TRAIN_X); % Update original training set accordingly
        end
        
        % Trim excess samples randomly if more than targetAugSamNum were generated
        while length(outTrain) > targetAugSamNum
            index = randi(length(outTrain));
            outTrain(index) = [];
            outTrainLabels(index) = [];
        end
        augTime = toc(tStart);
        
        % Append augmented data to original training data
        TRAIN_X = [TRAIN_X outTrain];
        TRAIN_Y = [TRAIN_Y outTrainLabels];
    else
        augTime = 0;
    end

    %% Classifier-specific parameter adjustments
    if strcmpi(params.classifier,'LSTM') || strcmpi(params.classifier,'GRU')
        % Number of classes
        params.numClasses = length(unique(trainingData(:,2)));
        % Hidden units heuristic: 3 × feature dimension
        params.numHiddenUnits = size(trainingData{1},2) * 3;
        
    elseif strcmpi(params.classifier,'Transformer')
        params.numClasses = length(unique(trainingData(:,2)));
        params.numHiddenUnits = size(trainingData{1},2) * 3;
        
        % Adjust numHiddenUnits to be divisible by numHeads squared for Transformer architecture
        modNumHiddenUnits = mod(params.numHiddenUnits, params.numHeads^2);
        if modNumHiddenUnits ~= 0
            params.numHiddenUnits = params.numHiddenUnits + (params.numHeads^2 - modNumHiddenUnits);
        end
        
        % Feed-forward network size typically 1.5× hidden units in Transformer
        params.feedForwardSize = round(params.numHiddenUnits * 1.5);
    end

    %% Perform classification test based on chosen classifier
    switch upper(params.classifier)
        case 'DTW'
            [accuracy, metrics] = test_dtw(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, params, ~multiprocessing);
        case 'LDMLT'
            [accuracy, metrics] = test_ldmlt(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, params, ~multiprocessing);
        case 'DE'
            [accuracy, metrics] = test_de(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, params, ~multiprocessing);
        case 'LSTM'
            [accuracy, metrics] = test_lstm(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, params, ~multiprocessing);
        case 'GRU'
            [accuracy, metrics] = test_gru(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, params, ~multiprocessing);
        otherwise % default to Transformer
            [accuracy, metrics] = test_transformer(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, params, ~multiprocessing);
    end
end
