%% Augmentation method:
% ARSPAWNER - Action Recognition Suboptimal Warped Time Series Generator
%
% Reference:
% D. Warchoł, M. Oszust, Augmentation of human action datasets with
% suboptimal warping and representative data samples, Sensors 22 (8), 2022.
%
% Description:
% This function augments multivariate time-series data by:
% 1. Selecting similar samples from the same class using Dynamic Time Warping (DTW),
% 2. Generating new samples by blending aligned segments from matched samples,
% 3. Validating generated samples against class prototypes using DTW to DBA representatives.
%
% Inputs:
% - train: cell array of multivariate time-series samples (dim × time)
% - trainLabels: vector of class labels for the training samples
% - nDraws: number of augmented versions to generate per sample
%
% Outputs:
% - outTrain: cell array with augmented training data
% - outTrainLabels: corresponding labels for the augmented data

function [outTrain, outTrainLabels] = aug_arspawner(train,trainLabels,nDraws)

outTrain = cell(1,length(trainLabels)*nDraws);
outTrainLabels = zeros(1,length(trainLabels)*nDraws);
xSize=size(train,2);
arrPod=zeros([xSize, xSize]);

% Compute DTW distance matrix between all pairs of samples
for i=1:1:xSize  
	tempI = train{i};
	for j=i+1:1:xSize  
		tempJ = train{j};
		window=ceil(size(tempI,2)/10);		 
		if size(tempI,2)< size(tempJ,2)
			window=ceil(size(tempJ,2)/10); 
		end			
		try 
			dt = dtw(tempI,tempJ,window)/(size(tempI,2)+size(tempJ,2));
			if isnan(dt)
				dt = dtw(tempI,tempJ)/(size(tempI,2)+size(tempJ,2));
			end
		catch ME
			dt=10e10;
		end
		arrPod(i,j) = dt; 
		arrPod(j,i) = dt;
	end 
end 

% Compute average DTW distance to same-class samples (per sample)
arrAvgClass=[]; 
for i=1:xSize 
	avgClass=0;
	tmpCount=0;
	for j=1:xSize 		
		if trainLabels(i) == trainLabels(j)
			avgClass=avgClass+arrPod(i,j);
			tmpCount=tmpCount+1;
		end
	end
	avgClass=avgClass/(tmpCount-1);
	arrAvgClass=[arrAvgClass;avgClass];
end

% Compute minimal DTW distance to same-class samples (tight similarity)
arrAvgClassMin=[]; 	
for i=1:xSize 
	minimum=1000e6; 			 
	for j=1:xSize 		
		if trainLabels(i) == trainLabels(j)
			if i~=j					
				if minimum > arrPod(i,j)
					minimum = arrPod(i,j);
				end
			end
		end
	end
	arrAvgClassMin=[arrAvgClassMin;minimum]; 			
end

% Compute per-class average DTW distances and standard deviations
arrClass=[]; 
arrClassSTD=[]; 
for n=1:max(trainLabels)
	avgClass=0;
	tmpCount=0;
	arrClassTmp=[]; 	
	for i=1:xSize 
		for j=i+1:xSize 		
			if trainLabels(j) == n 
				avgClass=avgClass+arrPod(i,j);
				arrClassTmp=[arrClassTmp;arrPod(i,j)];
				tmpCount=tmpCount+1;
			end
		end
	end
	avgClass=avgClass/(tmpCount-1);
	arrClass=[arrClass;avgClass];
	arrClassSTD=[arrClassSTD;std(arrClassTmp)];
end

% Compute DBA representatives for each class (used as class prototypes)
for n=min(trainLabels):max(trainLabels)
	tempCount=1; 
	for i=1:xSize 
        if trainLabels(i)==n			
			trainSet_n{tempCount}=train{i}; 	   
			tempCount=tempCount+1;
        end
	end
	if tempCount>1
		avgDBA{n}=DBAmult(trainSet_n); 
	end
end

jump=1;
counter=1;

% Main augmentation loop: generate new samples
while counter <size(outTrain,2)

checked=zeros([xSize,xSize]);

for i=1:jump:xSize   
	tempI = train{i};  
	odl4I=arrPod(i,:);
	[~,indy]= sort(odl4I,'asc');
	indy=indy(2:length(indy));
	tmpCount=0;
	for j=indy(1:length(indy)) 
		tempJ = train{j};
		if trainLabels(i) == trainLabels(j)   
			if checked(i,j)<=0
				checked(i,j)=checked(i,j)+1;
				checked(j,i)=checked(j,i)+1;
				for draw=1:nDraws
					randAB=rand();
					drawResA=size(tempI,2)*randAB;
					drawResB=size(tempJ,2)*randAB;
					drawResA=ceil(drawResA);
					drawResB=ceil(drawResB);

					% Split both samples into two parts
					t1a=tempI(:, 1:drawResA); 
					t1b=tempI(:, drawResA+1: size(tempI,2));     
					t2a=tempJ(:, 1:drawResB); 
					t2b=tempJ(:, drawResB+1: size(tempJ,2)); 			  

					% Align both parts independently using DTW
					ct1a= ceil(size(t1a,2)/10);
					ct2a= ceil(size(t2a,2)/10);
					ct1b= ceil(size(t1b,2)/10);
					ct2b= ceil(size(t2b,2)/10);

					window = max(ct1a,ct2a);
					if window<1, window=1; end
					try 
						[~,ix1,iy1] = dtw(t1a,t2a,window);
					catch ME
						continue
					end
					if size(t1a,2)>1 || size(t2a,2)>1
						onewarp1 = t1a(:,ix1);
						twowarp1 = t2a(:,iy1); 	
					else
						onewarp1 = t1a;
						twowarp1 = t2a;
					end

					window = max(ct1b,ct2b);
					if window<1, window=1; end
					[~,ix2,iy2] = dtw(t1b,t2b,window);
					if size(t1b,2)>1 || size(t2b,2)>1
						onewarp2 = t1b(:,ix2);
						twowarp2 = t2b(:,iy2); 	
					else
						onewarp2 = t1b;
						twowarp2 = t2b;
					end

					% Generate synthetic sample by adding noise to the average of aligned parts
					together1 = normrnd(0.5*(onewarp1+twowarp1),0.05*abs(onewarp1-twowarp1));
					together2 = normrnd(0.5*(onewarp2+twowarp2),0.05*abs(onewarp2-twowarp2));
					together = [together1,together2];  

					% Validate synthetic sample using DTW to class DBA prototype
					tempIp = avgDBA{trainLabels(i)};
					window = ceil(max(size(tempIp,2), size(together,2))/10);
					try 
						dt = dtw(tempIp,together,window)/(size(tempIp,2) + size(together,2));
					catch ME
						dt=10e10;
					end

					% Additional validation using original sample
					tempIp = train{i};  
					window = ceil(max(size(tempIp,2), size(together,2))/10);
					try 
						dt2 = dtw(tempIp,together,window)/(size(tempIp,2) + size(together,2));
					catch ME
						dt2=10e10;
					end

					% Apply class-specific thresholding rules
					avgClass = arrClass(trainLabels(j));
					partu_L = 0.25; 
					partu_R = 1.0;
					minClass = arrClass(trainLabels(j)) * partu_L + (arrClassSTD(trainLabels(j))*(partu_R+arrClassSTD(trainLabels(j))/avgClass));

					if dt < minClass && dt2 < minClass && dt > arrClass(trainLabels(j)) * partu_L && dt2 > arrClass(trainLabels(j)) * partu_L
						outTrainLabels(counter)=trainLabels(i); 
						outTrain{counter}=together; 	 
						counter=counter+1;
						tmpCount=tmpCount+1; 	
						if counter >= size(outTrain,2)
							break 
						end
					end 
				end
			end 
		end
	end 
end 

end 

end

% ---------------------- Support Functions ----------------------

function average = DBAmult(sequences)
    average = repmat(sequences{medoidIndex(sequences)},1);
	for i=1:15
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
