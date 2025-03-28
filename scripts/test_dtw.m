function accuracy = test_dtw(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, parameters, multiprocessing)
    if ~multiprocessing
        disp('k-NN DTW validation test...')
    end

    recognizedLabels = zeros(length(TEST_X), 1);
    recognizedSamplesCount = 0;
    
    for i = 1:length(TEST_X)    
        distances = zeros(length(TRAIN_X), 1);
        labels = zeros(length(TRAIN_X), 1);
        
        % Compute distances from the test sample to all training samples
        for j = 1:length(TRAIN_X)		 
            distances(j) = dtw(TEST_X{i}, TRAIN_X{j}, parameters.windowSize, parameters.metric);
            labels(j) = TRAIN_Y(j);
        end
        
        % Sort distances and get the k-nearest neighbors
        [~, sortedIndices] = sort(distances);
        kNearestLabels = labels(sortedIndices(1:parameters.k));
        
        % Determine the most common label among k-nearest neighbors
        recognizedLabel = mode(kNearestLabels);
        recognizedLabels(i) = recognizedLabel;
        
        if recognizedLabel == TEST_Y(i)
            recognizedSamplesCount = recognizedSamplesCount + 1;
        end
    end

    accuracy = recognizedSamplesCount / length(TEST_Y);
end