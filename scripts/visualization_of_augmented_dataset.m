% % Script for visualization of data samples as 2D embeddings
% % The dataset is augmented and original and augmented samples are diplayed
% % to see augmentation problems (e.g., ovelpapping classes, lack of diversity)

function visualization_of_augmented_dataset(augFunction, dataset, augSetSize)
    % Set random seed for reproducibility
    randn('seed', 0);
    rand('seed', 0);

    % Load dataset based on the provided name
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

    if isfield(datasetFiles, lower(dataset))
        data = importdata(datasetFiles.(lower(dataset)));
    else
        error('Unknown dataset name.');
    end

	subjectsNumber = length(data);
	Data = {};	
	for s = 1 : 1 : subjectsNumber
		Data = cat(1, Data, data{s});		
	end
    data=Data;

    % Determine the target number of augmented samples
    targetAugSamNum = augSetSize * length(data);

    % Extract features and labels
    numOriginalSamples = length(data);
    data_X = cell(1, numOriginalSamples);
    data_Y = zeros(1, numOriginalSamples);

    for i = 1:numOriginalSamples
        data_X{i} = data{i, 1}';
        data_Y(i) = str2double(data{i, 2});
    end
	
    % Data augmentation
    if isa(augFunction, 'function_handle')
        OutTrain = {};
        OutTrainLabels = [];

        while length(OutTrain) < targetAugSamNum
            [out_temp, out_lab_temp] = augment(data_X, data_Y, augFunction, 1);
            OutTrain = [OutTrain, out_temp];
            OutTrainLabels = [OutTrainLabels, out_lab_temp];
        end

        % Limit augmented samples to targetAugSamNum
        while length(OutTrain) > targetAugSamNum
            index = randi(length(OutTrain));
            OutTrain(index) = [];
            OutTrainLabels(index) = [];
        end

        % Merge original and augmented data
		if strcmp(func2str(augFunction), 'aug_adder') % This method is different, now we can change it
			data_X = change_representation(data_X);
		end
        data_X = [data_X, OutTrain];
        data_Y = [data_Y, OutTrainLabels];
    end

    % Prepare data for visualization
    AllData = data_X;
    AllLabels = data_Y;
    numSamples = length(AllData);

    % Compute DTW-based distance matrix
    D = zeros(numSamples, numSamples);
    for i = 1:numSamples - 1
        for j = i + 1:numSamples
            D(i, j) = dtw(AllData{i}, AllData{j}, 5, 'euclidean');
            D(j, i) = D(i, j);
        end
    end

    % Apply multidimensional scaling (MDS) to reduce dimensionality
    Y_all = mdscale(D, 2, 'Start', 'random');

    % Visualization
    figure;
    hold on;

    % Define different markers for original and augmented samples
    originalMarker = 'o'; % Circle for original data
    augmentedMarker = 'x'; % Cross for augmented data

    uniqueLabels = unique(AllLabels);
    colors = lines(length(uniqueLabels)); % Generate distinguishable colors
    legendEntries = gobjects(length(uniqueLabels) * 2, 1); % Placeholder for legend

    legendNames = cell(length(uniqueLabels) * 2, 1); % Labels for legend

    for i = 1:length(uniqueLabels)
        classLabel = uniqueLabels(i);

        % Find indices of original and augmented samples
        originalIdx = find(AllLabels(1:numOriginalSamples) == classLabel);
        augmentedIdx = find(AllLabels(numOriginalSamples + 1:end) == classLabel) + numOriginalSamples;

        % Plot original samples
        legendEntries(2 * i - 1) = scatter(Y_all(originalIdx, 1), Y_all(originalIdx, 2), 50, colors(i, :), originalMarker, 'filled');
        legendNames{2 * i - 1} = sprintf('Class %d (Original)', classLabel);

        % Plot augmented samples
        legendEntries(2 * i) = scatter(Y_all(augmentedIdx, 1), Y_all(augmentedIdx, 2), 50, colors(i, :), augmentedMarker);
        legendNames{2 * i} = sprintf('Class %d (Augmented)', classLabel);
    end

    % Set title with dataset name and augmentation method (bold)
	methodName = func2str(augFunction);
	if startsWith(methodName, 'aug_')
		methodName = methodName(5:end); % Delete "aug_"
	end
	methodName = upper(methodName); % Large letters

	title({sprintf('MDS - %s dataset', dataset), sprintf('\\bf{Augmentation: %s}', methodName)}, 'Interpreter', 'tex');
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    legend(legendEntries, legendNames);
    hold off;
end

 