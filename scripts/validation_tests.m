%% Validation Tests for Augmented Time Series Classification
%
% This script runs classification tests using a selected classifier on a specified
% time series dataset. Optionally, it applies data augmentation and performs multiple
% test repetitions to assess model robustness under randomized augmentation scenarios.
%
% Inputs:
% - augFunction: function handle for data augmentation (or string 'no aug')
% - classifier: name of the classifier to use (e.g., 'lstm', 'ldmlt')
% - dataset: string indicating the dataset name (e.g., 'auslan', 'msra', 'kard')
% - repetitions: number of times the test should be repeated
% - parameters: structure with classifier-specific hyperparameters; must include:
%       .classifier - repeated here for consistency
% - augSetSize: relative size of the augmented data (1 = 100% of original training size)
% - multiprocessing: boolean flag; if true, disables logs (for use in parallel jobs)
%
% Output:
% - accuracies: array of test accuracies across repetitions (or single value if no repetition)

function accuracies = validation_tests(augFunction, classifier, dataset, repetitions, parameters, augSetSize, multiprocessing)

    if nargin < 7
        multiprocessing = false;
    end

    % Set hardcoded randomization seeds for reproducibility
    randn('seed', 0);
    rand('seed', 0);

    % Load dataset
    switch lower(dataset)
        case 'florence'
            data = importdata('actionCoordsFLORENCE.mat');
        case 'kard'
            data = importdata('actionCoordsKARD.mat');
        case 'msra'
            data = importdata('actionCoordsMSRA.mat');
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

    % Split into training and testing sets (odd subjects for training)
    subjectsNumber = length(data);
    trainingData = {};
    testingData = {};   
    for s = 1 : 2 : subjectsNumber-1
        trainingData = cat(1, trainingData, data{s});
        testingData = cat(1, testingData, data{s+1});
    end

    % Determine number of augmented samples to generate
    targetAugSamNum = augSetSize * length(trainingData);

    parameters.classifier = classifier;

    % Execute test loop (repetitions > 1 only if augFunction is stochastic)
    if isa(augFunction, 'function_handle')
        accuracies = [];
        for i = 1:repetitions
            if ~multiprocessing
                disp(['Repetition number: ', int2str(i), '/', int2str(repetitions)])
            end
            if multiprocessing
                result = test(trainingData, testingData, augFunction, targetAugSamNum, parameters, multiprocessing);
            else
                result = test(trainingData, testingData, augFunction, targetAugSamNum, parameters, multiprocessing)
            end
            accuracies = [accuracies; result];
        end

        meanRates = mean(accuracies);
        stdRates = std(accuracies);
        if ~multiprocessing
            disp(['Mean accuracy: ', num2str(meanRates), ' Standard deviation: ', num2str(stdRates)])
        end
    else
        % No augmentation: single repetition
        if multiprocessing
            accuracies = test(trainingData, testingData, augFunction, targetAugSamNum, parameters, multiprocessing);
        else
            accuracies = test(trainingData, testingData, augFunction, targetAugSamNum, parameters, multiprocessing)
        end
    end
end
