% MW - Magnitude Warping

% T. T. Um, F. M. J. Pfister, D. Pichler, S. Endo, M. Lang, S. Hirche,
% U. Fietzek, D. Kuli´c, Data augmentation of wearable sensor data for
% parkinson’s disease monitoring using convolutional neural networks, in:
% ACM International Conference on Multimodal Interaction, ACM, ACM,
% 2017.

function [outTrain, outTrainLabels] = aug_mw(train,trainLabels,nDraws,num_knots,warp_std_dev)
% train - multivariate training set of time-series
% trainLabels - training set labels
% nDaws - number of augmentation epochs
% param num_knots - number of control points for splines
% param warp_std_dev - standard deviation for distorting the values of control points

%default values
if nargin < 4
    num_knots = 2;
end
if nargin < 5
    warp_std_dev = 0.1;
end

outTrain = cell(1,length(trainLabels)*nDraws);
outTrainLabels = zeros(1,length(trainLabels)*nDraws);
xSize = size(train,2);

counter = 1;
for i = 1 : xSize	 
	for iD = 1 : nDraws
		temp = train{i};

        if size(temp,2) < 2
            outTrain{counter} = temp;
		    outTrainLabels(counter) = trainLabels(i);
            counter = counter + 1;
            continue
        end

        [num_channels, num_time_steps] = size(temp);
        
        % Generate knot positions
        knot_positions = linspace(1, num_time_steps, num_knots);
        
        % Generate random distortions for knots (multiplied by 1 to keep variations centered)
        knot_values = 1 + warp_std_dev * randn(1, num_knots);
        
        % Generate time indices for the time series
        time_indexes = 1:num_time_steps;
        
        % Compute the warping function for each channel
        warped_series = zeros(size(temp));
        for ch = 1:num_channels
            % Fit a cubic spline to the knots
            spline_func = spline(knot_positions, knot_values, time_indexes);
            
            % Apply magnitude warping
            warped_series(ch, :) = temp(ch, :) .* spline_func;
        end
		
		outTrain{counter} = warped_series;
		outTrainLabels(counter) = trainLabels(i);

		counter = counter + 1;		
	end	
end



