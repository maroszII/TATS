function accuracy = one_fold_validation_dtw(trainingData, testingData, windowSize)
	recognizedSamplesCount = 0;
	for i = 1:length(testingData)	
		for j = 1:length(trainingData)
			distance = dtw(testingData{i,1}', trainingData{j,1}', windowSize, 'euclidean'); %euclidean distance
			if j==1 || distance < minDistance
				minDistance = distance;
				recognizedLabel = trainingData{j,2};
			end
		end
		if recognizedLabel == testingData{i,2}
			recognizedSamplesCount = recognizedSamplesCount + 1;
		end
		%disp(['i=' int2str(i)]);
	end
	accuracy = recognizedSamplesCount/length(testingData);
end