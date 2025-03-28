% Run experiments with data augmentation

if strcmpi(classifier,'LSTM') && strcmp(parameters.processingUnit,'gpu')
    multiprocessing = false;
    disp('LSTM with GPU is chosen. Multiprocessing will not be used.')
end

% Variable for storing results
numMethods = length(augmentationMethods);
results = zeros(numMethods, repetitions);

function updateProgress(total)
    persistent count;
    if isempty(count)
        count = 0;
    end
    count = count + 1;
    disp([int2str(count) '/' int2str(total) ' methods tested.']);
end

% Testing each augmentation method
if multiprocessing
    % Create worker processes
    parpool(numMethods);
    
    % Create a DataQueue for progress tracking
    q = parallel.pool.DataQueue;
    
    % Define persistent counter in callback
    afterEach(q, @(~) updateProgress(numMethods));
    
    % Preallocate results matrix
    results = zeros(numMethods, repetitions); % Assuming `repetitions` is defined
    
    parfor i = 1:numMethods
        augFunction = augmentationMethods{i};
        
        % Perform validation tests and store results
        accuracies = validation_tests(augFunction, classifier, dataset, repetitions, parameters, augSetSize, multiprocessing);
        results(i, :) = accuracies;
        
        % Send progress update
        send(q, i);
    end
    poolObj = gcp('nocreate'); 
    if ~isempty(poolObj)
        delete(poolObj);
    end
    disp('All methods tested.');
    results
else
    for i = 1:numMethods
        augFunction = augmentationMethods{i};
	    if isa(augFunction, 'function_handle')
		    disp(['Testing augmentation method: ', func2str(augFunction)]);
	    else
		    disp(['Testing augmentation method: ', augFunction]);
	    end
        
        % Perform validation tests and store results
        accuracies = validation_tests(augFunction, classifier, dataset, repetitions, parameters, augSetSize, multiprocessing);
        results(i, :) = accuracies;
    end
    disp('All methods tested.');
    results
end

% Save results to a file with a timestamp
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = ['Results_' timestamp '.mat'];
save(filename);


