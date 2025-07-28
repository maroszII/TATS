%% Visualization of Original and Augmented Dataset Using MDS Embedding
%
% This script visualizes the distribution of samples before and after data augmentation,
% using Multidimensional Scaling (MDS) based on DTW distance. The goal is to reveal issues
% such as class overlap or insufficient diversity introduced by augmentation.
%
% Inputs:
% - augFunction: function handle for augmentation method (e.g., @aug_dba, @aug_warp, etc.);
%                use 'no aug' to skip augmentation.
% - dataset: string specifying the dataset name (e.g., 'florence', 'kard', 'eeg').
% - augSetSize: multiplier of the original training set (e.g., 1.0 = same size as original).
%
% The script performs the following steps:
% 1. Loads the specified dataset and reformats the data.
% 2. Applies the selected augmentation method to produce additional training samples.
% 3. Computes the DTW-based distance matrix.
% 4. Reduces dimensionality via MDS.
% 5. Plots original and augmented samples for visual inspection.

function visualization_of_augmented_dataset(augFunction, dataset, augSetSize)

    %%% Set random seed for reproducibility
    randn('seed', 0);
    rand('seed', 0);

    %%% Map dataset names to their corresponding .mat files
    datasetFiles = struct( ...
        'florence', 'actionCoordsFLORENCE.mat', ...
        'kard', 'actionCoordsKARD.mat', ...
        'msra', 'actionCoordsMSRA.mat', ...
        'sysu', 'actionCoordsSYSU.mat', ...
        'utd', 'actionCoordsUTD.mat', ...
        'utk', 'actionCoordsUTK.mat', ...
        'visapp', 'actionCoordsVISAPP.mat', ...
        'arem', 'AReM.mat', ...
        'auslan', 'AUSLAN.mat', ...
        'ecg', 'ECG.mat', ...
        'eeg', 'EEG.mat', ...
        'gesturephasedetect', 'GesturePhaseDetect.mat', ...
        'kickvspunch', 'KickVsPunch.mat', ...
        'libras', 'LIBRAS.mat', ...
        'movementaal', 'MovementAAL.mat', ...
        'occupancydetect', 'OccupancyDetect.mat', ...
        'ozone', 'Ozone.mat', ...
        'pendigits', 'Pendigits.mat' ...
    );

    %%% Load selected dataset
    if isfield(datasetFiles, lower(dataset))
        data = importdata(datasetFiles.(lower(dataset)));
    else
        error('Unknown dataset name.');
    end

    %%% Flatten the data structure to a cell array
    subjectsNumber = length(data);
    Data = {};
    for s = 1 : subjectsNumber
        Data = cat(1, Data, data{s});
    end
    data = Data;

    %%% Determine number of augmented samples to generate
    targetAugSamNum = augSetSize * length(data);

    %%% Extract sequences (X) and labels (Y) from dataset
    numOriginalSamples = length(data);
    data_X = cell(1, numOriginalSamples);
    data_Y = zeros(1, numOriginalSamples);
    for i = 1:numOriginalSamples
        data_X{i} = data{i, 1}';
        data_Y(i) = str2double(data{i, 2});
    end

    %%% Augment data using the provided function handle
    if isa(augFunction, 'function_handle')
        OutTrain = {};
        OutTrainLabels = [];

        while length(OutTrain) < targetAugSamNum
            [out_temp, out_lab_temp] = augment(data_X, data_Y, augFunction, 1);
            OutTrain = [OutTrain, out_temp];
            OutTrainLabels = [OutTrainLabels, out_lab_temp];
        end

        % Truncate augmented data if too long
        while length(OutTrain) > targetAugSamNum
            index = randi(length(OutTrain));
            OutTrain(index) = [];
            OutTrainLabels(index) = [];
        end

        %%% For selected methods, convert representation before appending
        if strcmp(func2str(augFunction), 'aug_adder')
            data_X = change_representation(data_X);
        end

        % Merge original and augmented data
        data_X = [data_X, OutTrain];
        data_Y = [data_Y, OutTrainLabels];
    end

    %%% Prepare for distance matrix computation
    AllData = data_X;
    AllLabels = data_Y;
    numSamples = length(AllData);

    %%% Compute DTW distance matrix (quadratic cost)
    D = zeros(numSamples, numSamples);
    for i = 1:numSamples - 1
        for j = i + 1:numSamples
            D(i, j) = dtw(AllData{i}, AllData{j}, 5, 'euclidean');
            D(j, i) = D(i, j);
        end
    end

    %%% Apply MDS to project data into 2D space
    Y_all = mdscale(D, 2, 'Start', 'random');

    %%% Plot results
    figure;
    hold on;

    % Use circles for original samples, crosses for augmented
    originalMarker = 'o';
    augmentedMarker = 'x';

    uniqueLabels = unique(AllLabels);
    colors = lines(length(uniqueLabels));
    legendEntries = gobjects(length(uniqueLabels) * 2, 1);
    legendNames = cell(length(uniqueLabels) * 2, 1);

    for i = 1:length(uniqueLabels)
        classLabel = uniqueLabels(i);
        originalIdx = find(AllLabels(1:numOriginalSamples) == classLabel);
        augmentedIdx = find(AllLabels(numOriginalSamples + 1:end) == classLabel) + numOriginalSamples;

        legendEntries(2 * i - 1) = scatter(Y_all(originalIdx, 1), Y_all(originalIdx, 2), 50, colors(i, :), originalMarker, 'filled');
        legendNames{2 * i - 1} = sprintf('Class %d (Original)', classLabel);

        legendEntries(2 * i) = scatter(Y_all(augmentedIdx, 1), Y_all(augmentedIdx, 2), 50, colors(i, :), augmentedMarker);
        legendNames{2 * i} = sprintf('Class %d (Augmented)', classLabel);
    end

    %%% Format method name for title
    methodName = func2str(augFunction);
    if startsWith(methodName, 'aug_')
        methodName = methodName(5:end);
    end
    methodName = upper(methodName);

    %%% Show plot
    title({sprintf('MDS - %s dataset', dataset), sprintf('\\bf{Augmentation: %s}', methodName)}, 'Interpreter', 'tex');
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    legend(legendEntries, legendNames);
    hold off;

end
