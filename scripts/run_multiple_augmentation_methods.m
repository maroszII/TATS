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
% - accuracies: matrix of accuracies (methods × repetitions)
% - metrics: cell array of tables with other metrics (for each methods, averaged across all repetitions)
% - augTimes: augmentation times (in seconds)
% - saves results with timestamp in a MAT-file

if (strcmpi(classifier,'LSTM') || strcmpi(classifier,'GRU') || strcmpi(classifier,'Transformer')) && strcmp(parameters.processingUnit,'gpu')
    multiprocessing = false;
    disp('Deep learning classifier with GPU is chosen. Multiprocessing will not be used.')
end

numMethods = length(augmentationMethods);
accuracies = zeros(numMethods, parameters.repetitions); % Preallocate results matrix
metrics = cell(numMethods,2);
augTimes = zeros(numMethods, 1);

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
        [accuracies_val, metrics_val, augTime_val] = validation_tests(augFunction, classifier, dataset, parameters, multiprocessing);
        accuracies(i, :) = accuracies_val;
        metrics{i,2} = metrics_val;
        augTimes(i) = augTime_val;
        
        % Notify progress queue to update progress display
        send(q, i);
    end

    % Clean up parallel pool after completion
    poolObj = gcp('nocreate'); 
    if ~isempty(poolObj)
        delete(poolObj);
    end

    for i = 1:numMethods
        augFunction = augmentationMethods{i};

        if isa(augFunction, 'function_handle')
            metrics{i,1} = func2str(augFunction);
        else
            metrics{i,1} = augFunction;
        end
    end
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
        [accuracies_val, metrics_val, augTime_val] = validation_tests(augFunction, classifier, dataset, parameters, multiprocessing);
        accuracies(i, :) = accuracies_val;
        metrics{i,2} = metrics_val;
        augTimes(i) = augTime_val;
        if isa(augFunction, 'function_handle')
            metrics{i,1} = func2str(augFunction);
        else
            metrics{i,1} = augFunction;
        end
    end
end

disp('All methods tested.');
disp('Augmentation times for each metod:');
disp(augTimes);
disp('Accuracies for each metod (row) and each repetition (column):');
disp(accuracies);

% Save results with timestamp to MAT-file for later analysis
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = ['Results_' timestamp '.mat'];
save(filename, 'accuracies', 'metrics', 'augTimes', 'augmentationMethods', 'classifier', 'parameters');

% Nested function for progress display during multiprocessing
function updateProgress(total)
    persistent count;
    if isempty(count)
        count = 0;
    end
    count = count + 1;
    disp([int2str(count) '/' int2str(total) ' methods tested.']);
end

