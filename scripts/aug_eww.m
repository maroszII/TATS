% EWW - Extended Window Warping

% D. Warchoł, M. Oszust, Efficient augmentation of human action recog-
% nition datasets with warped windows, Procedia Computer Science 207
% (2022) 3018–3027, knowledge-Based and Intelligent Information En-
% gineering Systems: Proceedings of the 26th International Conference
% KES2022.

function [outTrain, outTrainLabels] = aug_eww(train,trainLabels,nDraws)
	% train - multivariate training set of time-series
	% trainLabels - training set labels
	% nDraws - number of augmentation epochs

	outTrain = cell(1,length(trainLabels)*nDraws);
    outTrainLabels = zeros(1,length(trainLabels)*nDraws);
	xSize = size(train,2);

	Ldiv = 3;

	counter = 1;
	for i = 1 : xSize
		for iD = 1 : nDraws
			temp = train{i};

            if size(temp,2) < 3
                outTrain{counter} = temp;
			    outTrainLabels(counter) = trainLabels(i);
                counter = counter + 1;
                continue
            end
			
			stretch_or_squeeze = randi([1 2]);
			 
			% FIRST EXCERPT - STRETCHING OR SQUEEZING
			if stretch_or_squeeze == 1
				% stretching
				nSamples = size(temp,2);	
			
				%1) randomly choose the length of the cutting and the place where we will attach it
				a = 0;	
				b = nSamples/Ldiv;
				excerpt = round((b-a).*rand()+ a);
				
				cutA = round(nSamples*rand());
				cutB = cutA+excerpt; 
				if cutB > nSamples
					cutB = nSamples;
				end
				if cutA == 0
					cutA = 1;
				end
				
				%2) cut
				excerpt = [];
				for j = cutA:cutB
					excerpt = [excerpt,j];
				end
				
				%3 non-linear stretch 
				excerpt2 = [];		
				for j = excerpt
					for drawRes = 1 : randi([0 1])
						excerpt2 = [excerpt2,j];
					end
				end
					
				for drawRes = 1 : nSamples	 
					excerpt2 = [excerpt2,drawRes];
				end
				
				excerpt = [excerpt2,excerpt];
				
				excerpt = sort(excerpt);
				temp = [temp(:,excerpt)];
			% END stretching
			else
				% squeezing
				nSamples = size(temp,2);
				a = 0;	
				b = nSamples/Ldiv;		
				wycinek = round((b-a).*rand()+ a);
				
				cutA = round(nSamples*rand());
				cutB = cutA+wycinek; 
				if cutB > nSamples
					cutB = nSamples;
				end
				if cutA ==0
					cutA=1;
				end
				
				%2) cut
				excerpt = [];
				for j = cutB : -3 : cutA
					excerpt = [excerpt,j];
				end
				
				% squeeze
				for j = excerpt
					drawRes = randi([1 2]);
					for L = 1:drawRes
						if j <= size(temp,2)
							temp(:,j) = [];
						end
					end
				end
			% END squeezing
			end
			% SECOND EXCERPT - STRETCHING OR SQUEEZING
			nSamples = size(temp,2);
			a = 0;	
			b = nSamples/Ldiv;		
			excerpt = round((b-a).*rand()+ a);
			
			cutA = round(nSamples*rand());
			if cutA == 0
				cutA=1;
			end
			
			cutB = cutA+excerpt; 
			if cutB > nSamples
				cutB = nSamples;
			end 
			
			subseq{1} = temp(:,cutA:cutB);
			subseqLen = size(subseq{1},2);
			
			incOrDec = randi([0 1]);
			if incOrDec == 0
				percentIncrease = 2.*rand() + 1; %random number from the range [1 - 3]
			else
				percentIncrease = 0.7.*rand() + 0.3; %random number from the range [0.3 - 1]
			end
					   
			newLen = round(percentIncrease*subseqLen);
			
			subseqInterpolated = interpolateXT(subseq, newLen);
			subseqInterpolated = subseqInterpolated{1};
			
			if(cutA > 1)
				tempLeft = temp(:, 1:cutA-1);
			else
				tempLeft = [];
			end
				
			if(cutB < nSamples)
				tempRight = temp(:, cutB+1:nSamples);
			else
				tempRight = [];
			end
			
			temp = [tempLeft subseqInterpolated tempRight];
			%END INTERPOLATION

			%3) adding results to the set
			outTrain{counter} = temp;
			outTrainLabels(counter) = trainLabels(i);
			
			counter = counter + 1;		
		end	
	end
end

function XT = interpolateXT(XTinput,newLen)
	XT = XTinput;

	for i=1:size(XT,1)
		temp = XT{i}';

		if size(temp,1)==1
			temp=[temp;temp];
		end

		[initSize1, initSize2] = ndgrid(1:size(temp, 1), 1:size(temp, 2));
		[newSize1, newSize2] = ndgrid(linspace(1, size(temp, 1), newLen), 1:size(temp, 2));
		newData = interpn(initSize1, initSize2, temp, newSize1, newSize2);

		XT{i}=newData';
	end
end
