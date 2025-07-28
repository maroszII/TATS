%% Augmentation method:
%% MW - Magnitude Warping
%
% T. T. Um, F. M. J. Pfister, D. Pichler, S. Endo, M. Lang, S. Hirche,
% U. Fietzek, D. Kulić, 
% Data augmentation of wearable sensor data for Parkinson’s disease monitoring using convolutional neural networks,
% ACM International Conference on Multimodal Interaction, ACM, 2017.
%
% This function augments multivariate time-series data by:
% 1. Generating smooth random magnitude warping curves using cubic splines,
% 2. Applying these warping curves to scale the original time-series values,
% 3. Producing augmented samples that mimic natural variability in sensor magnitude.
%
% Inputs:
% - train: cell array of multivariate time-series samples (dim × time)
% - trainLabels: vector of class labels corresponding to the training samples
% - nDraws: number of augmented samples to generate per original sample
% - num_knots: number of spline control points to define the warping curve (optional, default: 2)
% - warp_std_dev: standard deviation controlling the randomness of the warping curve (optional, default: 0.1)
%
% Outputs:
% - outTrain: cell array of augmented training samples
% - outTrainLabels: vector of corresponding class labels for augmented samples

function [outTrain, outTrainLabels] = aug_mw(train,trainLabels,nDraws,num_knots,warp_std_dev)

% Set default values if parameters are not provided
if nargin < 4
    num_knots = 2; % Number of spline knots to keep warping curve smooth and simple
end
if nargin < 5
    warp_std_dev = 0.1; % Standard deviation controlling warping intensity
end

outTrain = cell(1,length(trainLabels)*nDraws);
outTrainLabels = zeros(1,length(trainLabels)*nDraws);
xSize = size(train,2);

counter = 1;
for i = 1 : xSize
    for iD = 1 : nDraws
        temp = train{i};

        % Skip augmentation if time series is too short for meaningful warping
        if size(temp,2) < 2
            outTrain{counter} = temp;
            outTrainLabels(counter) = trainLabels(i);
            counter = counter + 1;
            continue
        end

        [num_channels, num_time_steps] = size(temp);

        % Generate knot positions evenly spaced along time axis
        knot_positions = linspace(1, num_time_steps, num_knots);

        % Generate random distortions for knots centered around 1
        knot_values = 1 + warp_std_dev * randn(1, num_knots);

        % Time indices for interpolation of the spline curve
        time_indexes = 1:num_time_steps;

        warped_series = zeros(size(temp));
        for ch = 1:num_channels
            % Create cubic spline warping curve for each channel
            spline_func = spline(knot_positions, knot_values, time_indexes);

            % Apply magnitude warping by elementwise multiplication
            warped_series(ch, :) = temp(ch, :) .* spline_func;
        end

        % Store augmented sample and corresponding label
        outTrain{counter} = warped_series;
        outTrainLabels(counter) = trainLabels(i);

        counter = counter + 1;
    end
end

end
