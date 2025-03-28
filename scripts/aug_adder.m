% Adder

% M. Oszust, D. Warchoł, Time series augmentation with time-scale
% modifications and piecewise aggregate approximation for human ac-
% tion recognition, in: 2022 IEEE 34th International Conference on
% Tools with Artificial Intelligence (ICTAI), 2022, pp. 700–704.

function [outTrain,outTrainLabels] = aug_adder(train,trainLabels,nDraws,segNum)
% train - multivariate training set of time-series
% trainLabels - training set labels
% nDraws - number of augmentation epochs
% segNum - number of segments

if nargin < 4
    segNum = 10;
end

outTrain = cell(1,length(trainLabels)*nDraws); 
outTrainLabels = [];
for i=1:nDraws
	outTrainLabels = [outTrainLabels trainLabels];
end

segNumOrig = segNum;

maxDl = 0;
minDl = 10e10;
for i=1:length(train)
	mtemp = length(train{i});
	if mtemp > maxDl
		maxDl = mtemp;
	end 
	if mtemp < minDl
		minDl = mtemp;
	end 
end	

maxDlM = ceil(maxDl*2 / segNum) * segNum;
minDlM = segNum;
a = minDlM:segNum:maxDlM;

counter = 1;
for lP=1:nDraws
	for i=1:length(train)
		tempI = train{i};

        if size(tempI,2) < 3
            outTrain{counter} = tempI;
		    outTrainLabels(counter) = trainLabels(i);
            counter = counter + 1;
            continue
        end

		 
			if lP>1
				segNum = ceil(segNumOrig*0.8):ceil(segNumOrig*1.2);
				segNum = segNum(randperm(length(segNum))); 
				segNum = segNum(1);
				maxDlM = ceil(maksDl*2 / segNum) * segNum;
				minDlM = segNum;
				a = minDlM:segNum:maxDlM;
			end

			data_len = size(tempI,2);
			a_rand = a(randperm(length(a)));
			aa = a_rand(1);
			data_len = ceil(data_len / aa) * aa;

			data = tempI';
			newLen = data_len;

			[initSize1, initSize2] = ndgrid(1:size(data, 1), 1:size(data, 2));
			[newSize1, newSize2] = ndgrid(linspace(1, size(data, 1), newLen), 1:size(data, 2));
			newData = interpn(initSize1, initSize2, data, newSize1, newSize2);

			segSiz = floor(data_len/segNum);

			tempR =[];
			for d=1:size(newData,2)
				data = newData(:,d);		
				data = (data - mean(data))/std(data);
				dane = [reshape(data,segSiz,segNum)]; 
				dane(isnan(dane)) = 0;
		        segments =  mean(dane);

				TF = isnan(segments);
				if TF
					segments(isnan(segments)) = 0;				
				end
				tempR = [tempR,segments'];
			end
			outR=tempR;	
		 
		outTrain{counter} = outR';
		counter = counter+1;
	end
 end
 
 
 