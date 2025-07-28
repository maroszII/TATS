%% Run experiments with data augmentation
%
% This script runs validation tests on multiple data augmentation methods.
% It supports parallel processing unless LSTM with GPU is selected, where
% multiprocessing is disabled due to incompatibility.
%
% Inputs (expected to be defined externally):
% - classifier: string, e.g., 'LSTM'
% - parameters.processingUnit: string, e.g., 'gpu' or 'cpu'
% - augmentationMethods: cell array of function handles or method names
% - repetitions: number of validation repetitions per method
% - dataset: dataset used for training/testing
% - parameters: struct with training/testing parameters
% - augSetSize: augmentation set size (integer)
% - multiprocessing: boolean flag to enable parallel processing
%
% Outputs:
% - results: matrix of accuracies (methods × repetitions)
% - saves results with timestamp in a MAT-file

if (strcmpi(classifier,'LSTM') || strcmpi(classifier,'GRU') || strcmpi(classifier,'Transformer')) && strcmp(parameters.processingUnit,'gpu')
    multiprocessing = false;
    disp('Deep learning classifier with GPU is chosen. Multiprocessing will not be used.')
end

numMethods = length(augmentationMethods);
results = zeros(numMethods, repetitions); % Preallocate results matrix

if multiprocessing
    % Initialize parallel pool with one worker per augmentation method
    parpool(numMethods);

    % DataQueue to handle progress updates from workers
    q = parallel.pool.DataQueue;
    afterEach(q, @(~) updateProgress(numMethods));

    % Run parallel loop over augmentation methods
    parfor i = 1:numMethods
        augFunction = augmentationMethods{i};
        
        % Run validation tests for the current augmentation method
        accuracies = validation_tests(augFunction, classifier, dataset, ...
            repetitions, parameters, augSetSize, multiprocessing);
        results(i, :) = accuracies;
        
        % Notify progress queue to update progress display
        send(q, i);
    end

    % Clean up parallel pool after completion
    poolObj = gcp('nocreate'); 
    if ~isempty(poolObj)
        delete(poolObj);
    end

    disp('All methods tested.');
    disp(results);

else
    % Sequential processing when multiprocessing disabled
    for i = 1:numMethods
        augFunction = augmentationMethods{i};
        
        % Display which augmentation method is currently tested
        if isa(augFunction, 'function_handle')
            disp(['Testing augmentation method: ', func2str(augFunction)]);
        else
            disp(['Testing augmentation method: ', augFunction]);
        end
        
        % Run validation tests for the current augmentation method
        accuracies = validation_tests(augFunction, classifier, dataset, ...
            repetitions, parameters, augSetSize, multiprocessing);
        results(i, :) = accuracies;
    end
    disp('All methods tested.');
    disp(results);
end

% Save results with timestamp to MAT-file for later analysis
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = ['Results_' timestamp '.mat'];
save(filename, 'results', 'augmentationMethods', 'classifier', 'parameters');

% Nested function for progress display during multiprocessing
function updateProgress(total)
    persistent count;
    if isempty(count)
        count = 0;
    end
    count = count + 1;
    disp([int2str(count) '/' int2str(total) ' methods tested.']);
end

