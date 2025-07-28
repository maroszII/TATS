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
% - parameters: struct containing classifier and training parameters
% - multiprocessing: boolean flag indicating if multiprocessing is enabled
%
% Output:
% - accuracy: classification accuracy on the test set

function accuracy = test(trainingData, testingData, augFunction, targetAugSamNum, parameters, multiprocessing)

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

    %% Data augmentation (if augmentation function provided)
    % Generate augmented data until target sample count is reached
    if isa(augFunction, 'function_handle')
        if ~multiprocessing
            disp('Augmenting training data...');
        end
        
        outTrain = [];
        outTrainLabels = [];
        
        % Repeat augmentation calls until enough augmented samples generated
        while true
            if strcmp(func2str(augFunction), 'aug_adder')
                % Special handling: aug_adder requires change in testing set representation
                [out_temp, out_lab_temp] = augment(TRAIN_X, TRAIN_Y, augFunction, 1); 
                TEST_X = change_representation(TEST_X); % Update test set accordingly
            else
                [out_temp, out_lab_temp] = augment(TRAIN_X, TRAIN_Y, augFunction, 1);
            end
            
            outTrain = [outTrain out_temp];
            outTrainLabels = [outTrainLabels out_lab_temp];
            
            if length(outTrain) >= targetAugSamNum
                break; % Stop augmenting once sufficient samples generated
            end
        end
        
        % Trim excess samples randomly if more than targetAugSamNum were generated
        while length(outTrain) > targetAugSamNum
            index = randi(length(outTrain));
            outTrain(index) = [];
            outTrainLabels(index) = [];
        end
        
        % Append augmented data to original training data
        TRAIN_X = [TRAIN_X outTrain];
        TRAIN_Y = [TRAIN_Y outTrainLabels];
    end

    %% Classifier-specific parameter adjustments
    
    if strcmpi(parameters.classifier,'LSTM') || strcmpi(parameters.classifier,'GRU')
        % Number of classes
        parameters.numClasses = length(unique(trainingData(:,2)));
        % Hidden units heuristic: 3 × feature dimension
        parameters.numHiddenUnits = size(trainingData{1},2) * 3;
        
    elseif strcmpi(parameters.classifier,'Transformer')
        parameters.numClasses = length(unique(trainingData(:,2)));
        parameters.numHiddenUnits = size(trainingData{1},2) * 3;
        
        % Adjust numHiddenUnits to be divisible by numHeads squared for Transformer architecture
        modNumHiddenUnits = mod(parameters.numHiddenUnits, parameters.numHeads^2);
        if modNumHiddenUnits ~= 0
            parameters.numHiddenUnits = parameters.numHiddenUnits + (parameters.numHeads^2 - modNumHiddenUnits);
        end
        
        % Feed-forward network size typically 1.5× hidden units in Transformer
        parameters.feedForwardSize = round(parameters.numHiddenUnits * 1.5);
    end

    %% Perform classification test based on chosen classifier
    
    switch upper(parameters.classifier)
        case 'DTW'
            accuracy = test_dtw(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, ~multiprocessing);
        case 'LDMLT'
            accuracy = test_ldmlt(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, ~multiprocessing);
        case 'DE'
            accuracy = test_de(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, ~multiprocessing);
        case 'LSTM'
            accuracy = test_lstm(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, ~multiprocessing);
        case 'GRU'
            accuracy = test_gru(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, ~multiprocessing);
        otherwise % default to Transformer
            accuracy = test_transformer(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, ~multiprocessing);
    end
end
