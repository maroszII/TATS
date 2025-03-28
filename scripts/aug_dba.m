% DBA - Dynamic Timer Warping Barycenter Averaging

% H. I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, P.-A. Muller, Data
% augmentation using synthetic data for time series classification with deep
% residual networks, CoRR abs/1808.02455 (2018).

function [outTrain, outTrainLabels] = aug_dba(train, trainLabels, nDraws)
	% train - Cell array containing the original time-series training set
	% trainLabels - Vector of training set labels
	% nDraws - Number of augmentation iterations per class 
    numSamples = length(trainLabels) * nDraws;
    outTrain = cell(1, numSamples);
    outTrainLabels = zeros(1, numSamples);

    counter = 1;
    
    % Iterate over the number of augmentation draws
    for iD = 1:nDraws                
		for classLabel=min(trainLabels):max(trainLabels) 
            % Select all training samples belonging to the current class
            trainingSetN = train(trainLabels == classLabel);

            % Ensure there are enough samples for selection
            numClassSamples = size(trainingSetN, 2);
            if numClassSamples < 2
                continue; % Skip classes with fewer than 2 samples
            end

            % Randomly select a subset of samples
            randOrder = randperm(numClassSamples);
            numToSelect = randi([1, numClassSamples]);  
            selectedIndices = randOrder(1:numToSelect);
            trainingSubset = trainingSetN(selectedIndices);

            % Apply DBA only if the subset is non-empty
            if ~isempty(trainingSubset)
                outTrainLabels(counter) = classLabel;
                outTrain{counter} = DBAmult(trainingSubset);
                counter = counter + 1;
            end
        end
    end
    % Remove unused preallocated space (if any)
    outTrain(counter:end) = [];
    outTrainLabels(counter:end) = [];
end
 

function average = DBAmult(sequences)
    average = repmat(sequences{medoidIndex(sequences)},1);
	for i=1:15 % Additional parameter. 
		average = DBA_one_iterationMult(average,sequences);
	end
end

function sos = sumOfSquares(s,sequences)
    sos = 0.0;
    for i=1:size(sequences,2)
        dist = dtw(s,sequences{i});
        sos = sos + dist * dist;
    end
end

function index = medoidIndex(sequences) 
	index = -1;
    lowestInertia = Inf;
    for i=1:size(sequences,2)
        tmpInertia = sumOfSquares(sequences{i},sequences);
        if (tmpInertia < lowestInertia)
            index = i;
            lowestInertia = tmpInertia;
        end
    end
end

function average = DBA_one_iterationMult(averageS,sequences)
    tupleAssociation = cell(size(averageS,1), size(averageS,2));

	for p=1:size(averageS,1)
		for t=1:size(averageS,2)
			tupleAssociation{p,t} = [];
		end
	end
	
	costMatrix = zeros(1000,1000);
	pathMatrix = zeros(1000,1000);
	
	if size(averageS,2)>1000
		disp('DBA: Too small matrix vectors.')
	end

	for k=1:size(sequences,2)
	    sequence = sequences{k};
	    costMatrix(1,1) = distanceTo(averageS(:,1),sequence(:,1));
	    pathMatrix(1,1) = -1;
	    for i=2:size(averageS,2)
			costMatrix(i,1) = costMatrix(i-1,1) + distanceTo(averageS(:,i),sequence(:,1));
			pathMatrix(i,1) = 2;
	    end
	    
	    for j=2:size(sequence,2)
			costMatrix(1,j) = costMatrix(1,j-1) + distanceTo(sequence(:,j),averageS(:,1));
			pathMatrix(1,j) = 1;
	    end
	    
	    for i=2:size(averageS,2)
			for j=2:size(sequence,2)
				indiceRes = ArgMin3(costMatrix(i-1,j-1),costMatrix(i,j-1),costMatrix(i-1,j));
				pathMatrix(i,j) = indiceRes;
				
				if indiceRes==0
					res = costMatrix(i-1,j-1);
				elseif indiceRes==1
					res = costMatrix(i,j-1);
				elseif indiceRes==2
					res = costMatrix(i-1,j);
				end
				
				costMatrix(i,j) = res + distanceTo(averageS(:,i),sequence(:,j));
			end
	    end

		for dimension = 1 : size(averageS,1) 
			i=size(averageS,2);
			j=size(sequence,2);
			while(true)
				tupleAssociation{dimension,i}(end+1) = sequence(dimension,j);   
				if pathMatrix(i,j)==0
					i=i-1;
					j=j-1;
				elseif pathMatrix(i,j)==1
					j=j-1;
				elseif pathMatrix(i,j)==2
					i=i-1;          
				else
					break
				end
			end
		end
	end

	for dimension = 1 : size(averageS,1) 
		for t = 1 : size(averageS,2)
		   averageS(dimension,t) = mean(tupleAssociation{dimension,t});	   
		end
	end  
	average = averageS;
end

function value = ArgMin3(a,b,c)
	if (a<b)
	    if (a<c)
			value=0;
			return
	    else
			value=2;
			return
	    end
	else
	    if (b<c)
			value=1;
			return
	    else
			value=2;
			return
	    end
	end
end

function dist = distanceTo(a,b)
	dist = sum((a - b).^ 2);
end