% WW - Window Warping

% A. Le Guennec, S. Malinowski, R. Tavenard, Data augmentation
% for time series classification using convolutional neural networks, in:
% ECML/PKDD Workshop on Advanced Analytics and Learning on Tem-
% poral Data, Riva Del Garda, Italy, 2016.

function [outTrain, outTrainLabels] = aug_ww(train,trainLabels,nDraws)
% train - multivariate training set of time-series
% trainLabels - training set labels
% nDaws - number of augmentation epochs

outTrain = cell(1,length(trainLabels)*nDraws);
outTrainLabels = zeros(1,length(trainLabels)*nDraws);
xSize = size(train,2);

counter = 1;
for i = 1 : xSize
	for iD = 1 : nDraws
		temp = train{i};
		nSamples = size(train{i},2);	
		
		a = 0;	
		b = nSamples/3;
		excerpt = round((b-a).*rand() + a);
		
		cutA = round(nSamples*rand());
		cutB = cutA + excerpt; 
		if cutB > nSamples
			cutB = nSamples;
		end
		if cutA == 0
			cutA = 1;
		end

		excerpt = [];
		for j = cutA:cutB
			excerpt = [excerpt,j];
		end

		excerpt2 = [];		
		for j = excerpt
			for los = 1:round((2-1).*rand() + 1)
				excerpt2 = [excerpt2,j];
			end
		end

		for los = 1 : nSamples	 
			excerpt2 = [excerpt2,los];
		end
		
		excerpt = [excerpt2,excerpt];
		excerpt = sort(excerpt);
		temp = [temp(:,excerpt)];

		outTrain{counter} = temp;
		outTrainLabels(counter) = trainLabels(i);

		counter = counter + 1;		
	end
end


