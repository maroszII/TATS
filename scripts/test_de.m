%% Delay Embedding Classifier Validation Test
%
% This function performs classification using the Delay Embedding (DE) method.
% The training data is converted to transition probability models for each class,
% then test samples are classified by comparing their embeddings to these models.
%
% Inputs:
% - TRAIN_X: cell array of training samples (features × time)
% - TRAIN_Y: vector of training labels
% - TEST_X: cell array of testing samples
% - TEST_Y: vector of testing labels
% - parameters: struct with classifier parameters (gridSize, dim, step, slide, filter, alpha, beta, isGrid)
% - dispLogs: boolean flag to display progress messages
%
% Output:
% - accuracy: classification accuracy on the test set

function accuracy = test_de(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, dispLogs)

    %% Add necessary paths for DE functions
 	addpath('scripts/DE/DE', 'scripts/DE/MGM', 'scripts/DE/utilities');
	
	% display logs only if multiprocessing is set
    if dispLogs
        disp('Delay Embedding validation test...')
    end
	
	% create grid for each class
	classLabels = unique(TEST_Y);
	n_class = length(classLabels);
	n_dimSignal = size(TRAIN_X{1},1);
	trans = cell(n_class, 1);
	grid = cell(n_class, 1);
	for i=1:n_class
		grid{i} = createGrid(parameters.gridSize, zeros(1, parameters.dim*n_dimSignal));
	end

	% training
	for loop = 1:length(TRAIN_X)
		% extract data and label
		x = TRAIN_X{loop};
		% low-pass filter
		for i = 1:size(x, 1)
			x(i, :) = lowpassFilter(x(i,:), parameters.filter);
		end
		y = TRAIN_Y(loop);
        if(size(x,2) < parameters.dim)
            continue
        end
		% multi-dimensional delay embedding
		point_cloud = delayEmbedingND(x', parameters.dim, parameters.step, parameters.slide);
		% update transition list
		trans{y} = add2Trans(point_cloud, trans{y}, grid{y}, parameters.isGrid);
	end
	% refine transition list and compute transition probability
	for i=1:n_class
		trans{i} = Trans_Prob(trans{i});
	end 

	% testing
	dist = zeros(n_class, 1);
	prediction = zeros(length(TEST_X), 1);
	for loop = 1:length(TEST_X)
		% extract data and label
		x = TEST_X{loop};
		% low-pass filter
        for i = 1:size(x, 1)
			x(i, :) = lowpassFilter(x(i,:), parameters.filter);
        end
		% multi-dimensional delay embedding
		point_cloud = delayEmbedingND(x', parameters.dim, parameters.step, parameters.slide);
		% model matching
		for i = 1:n_class
			dist(i) = HDist(point_cloud, trans{i}, grid{i}, parameters.alpha, parameters.beta, parameters.isGrid);
		end 
		[~, loc] = min(dist);
		prediction(loop) = loc; 
	end
	
	% calculate classification accuracy
	accuracy = mean(TEST_Y'==prediction);
end