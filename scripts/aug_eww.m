%
% Augmentation method: Extended Window Warping (EWW)
%
% This function implements the EWW technique for augmenting multivariate time-series data.
% The method is described in:
% D. Warchoł, M. Oszust, Efficient augmentation of human action recog-
% nition datasets with warped windows, Procedia Computer Science 207 (2022) 3018–3027.
% It modifies local windows of the signal by:
% 1. Randomly stretching or squeezing internal windows (simulating timing variations),
% 2. Interpolating a second random window with varied scaling,
% 3. Concatenating results back into the signal.
%
% Inputs:
% - train: cell array of multivariate time-series samples (dim × time)
% - trainLabels: corresponding class labels
% - nDraws: number of augmented versions to generate per sample
%
% Outputs:
% - outTrain: augmented training data
% - outTrainLabels: corresponding labels for the augmented data

function [outTrain, outTrainLabels] = aug_eww(train,trainLabels,nDraws)
	% Allocate output arrays
	outTrain = cell(1,length(trainLabels)*nDraws);
    outTrainLabels = zeros(1,length(trainLabels)*nDraws);
	xSize = size(train,2);
	Ldiv = 3; % Used to define relative excerpt sizes

	counter = 1;
	for i = 1 : xSize
		for iD = 1 : nDraws
			temp = train{i};

            if size(temp,2) < 3
                % Too short to modify — copy directly
                outTrain{counter} = temp;
			    outTrainLabels(counter) = trainLabels(i);
                counter = counter + 1;
                continue
            end
			
			stretch_or_squeeze = randi([1 2]);
			
			% === FIRST EXCERPT — Stretching or Squeezing ===
			if stretch_or_squeeze == 1
				% STRETCHING: repeat some points locally
				nSamples = size(temp,2);
				a = 0;
				b = nSamples/Ldiv;
				excerpt = round((b-a).*rand()+ a);
				
				cutA = round(nSamples*rand());
				cutB = cutA+excerpt;
				if cutB > nSamples, cutB = nSamples; end
				if cutA == 0, cutA = 1; end
				
				% Build index list of repeated samples
				excerpt = cutA:cutB;
				excerpt2 = [];
				for j = excerpt
					for drawRes = 1:randi([0 1])
						excerpt2 = [excerpt2, j];
					end
				end
				% Append original indices to retain structure
				for drawRes = 1:nSamples
					excerpt2 = [excerpt2, drawRes];
				end
				excerpt = sort([excerpt2, excerpt]);
				temp = temp(:, excerpt);
			else
				% SQUEEZING: randomly remove points from a window
				nSamples = size(temp,2);
				a = 0;
				b = nSamples/Ldiv;
				wycinek = round((b-a).*rand()+ a);
				cutA = round(nSamples*rand());
				cutB = cutA+wycinek;
				if cutB > nSamples, cutB = nSamples; end
				if cutA == 0, cutA = 1; end

				excerpt = cutB:-3:cutA;
				for j = excerpt
					drawRes = randi([1 2]);
					for L = 1:drawRes
						if j <= size(temp,2)
							temp(:,j) = [];
						end
					end
				end
			end
			% === END FIRST MODIFICATION ===

			% === SECOND EXCERPT — Interpolation of a random subwindow ===
			nSamples = size(temp,2);
			a = 0;
			b = nSamples/Ldiv;
			excerpt = round((b-a).*rand()+ a);
			cutA = round(nSamples*rand());
			if cutA == 0, cutA=1; end
			cutB = cutA+excerpt;
			if cutB > nSamples, cutB = nSamples; end

			subseq{1} = temp(:,cutA:cutB);
			subseqLen = size(subseq{1},2);
			
			incOrDec = randi([0 1]);
			if incOrDec == 0
				percentIncrease = 2.*rand() + 1; % Range [1–3]
			else
				percentIncrease = 0.7.*rand() + 0.3; % Range [0.3–1]
			end

			newLen = round(percentIncrease * subseqLen);
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

			% Replace subwindow with interpolated one
			temp = [tempLeft subseqInterpolated tempRight];
			% === END INTERPOLATION ===

			outTrain{counter} = temp;
			outTrainLabels(counter) = trainLabels(i);
			counter = counter + 1;
		end
	end
end

function XT = interpolateXT(XTinput,newLen)
	% Performs interpolation to resize a given segment to newLen
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
