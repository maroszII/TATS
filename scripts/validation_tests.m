function accuracies = validation_tests(augFunction, classifier, dataset, repetitions, parameters, augSetSize, multiprocessing)

if nargin < 7
    multiprocessing = false;
end

randn('seed', 0);
rand('seed', 0);

switch lower(dataset)
    case 'florence'
        data = importdata('actionCoordsFLORENCE.mat');
    case 'kard'
        data = importdata('actionCoordsKARD.mat');
    case 'msra'
        data = importdata('actionCoordsMSRA.mat');
    case 'sysu'
        data = importdata('actionCoordsSYSU.mat');
    case 'utd'
        data = importdata('actionCoordsUTD.mat');
    case 'utk'
        data = importdata('actionCoordsUTK.mat');
    case 'visapp'
        data = importdata('actionCoordsVISAPP.mat');
    case 'arem'
        data = importdata('AReM.mat');
    case 'auslan'
        data = importdata('AUSLAN.mat');
    case 'ecg'
        data = importdata('ECG.mat');
    case 'eeg'
        data = importdata('EEG.mat');
    case 'gesturephasedetect'
        data = importdata('GesturePhaseDetect.mat');
    case 'kickvspunch'
        data = importdata('KickVsPunch.mat');
    case 'libras'
        data = importdata('LIBRAS.mat');
    case 'movementaal'
        data = importdata('MovementAAL.mat');
    case 'occupancydetect'
        data = importdata('OccupancyDetect.mat');
    case 'ozone'
        data = importdata('Ozone.mat');
    case 'pendigits'
        data = importdata('Pendigits.mat');
    otherwise
        error('Unknown dataset name.');
end

subjectsNumber = length(data);
trainingData = {};
testingData = {};   
for s = 1 : 2 : subjectsNumber-1
    trainingData = cat(1, trainingData, data{s});
    testingData = cat(1, testingData, data{s+1});
end

targetAugSamNum = augSetSize * length(trainingData); % Generate augSetSize times as many new samples as there are in the training set
 

if strcmpi(classifier,'LSTM')
    parameters.numClasses = length(unique(trainingData(:,2)));
    parameters.numHiddenUnits = size(trainingData{1},2) * 3;
end
parameters.classifier = classifier;

if isa(augFunction, 'function_handle')
    accuracies = [];
    % Repeat tests because augmentation methods have random elements
    for i=1:repetitions
        if ~multiprocessing
            disp(['Repetition number: ', int2str(i), '/', int2str(repetitions)])
        end
    
        if multiprocessing
            result = test(trainingData,testingData,augFunction,targetAugSamNum,parameters,multiprocessing);
        else
            result = test(trainingData,testingData,augFunction,targetAugSamNum,parameters,multiprocessing)
        end
    
        accuracies = [accuracies;result];
    end
    meanRates = mean(accuracies);
    stdRates = std(accuracies);
    if ~multiprocessing
	    disp(['Mean accuracy: ', num2str(meanRates), ' Standard deviation: ', num2str(stdRates)]) 
    end
else
    if multiprocessing
        accuracies = test(trainingData,testingData,augFunction,targetAugSamNum,parameters,multiprocessing);
    else
        accuracies = test(trainingData,testingData,augFunction,targetAugSamNum,parameters,multiprocessing)
    end
end
