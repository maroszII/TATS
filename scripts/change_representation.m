%% Change representation of time-series by segment averaging
%
% This function should be called after augmentation (e.g., after aug_adder)
% to extend and transform the time-series data of a testing set.
% It divides each sequence into a specified number of segments (segNum).
% Then it represents each segment by averaging feature values, effectively
% downsampling and smoothing the data in the feature dimension.
%
% Inputs:
% - set: cell array of multivariate time-series samples (features × time)
% - segNum: number of segments to divide each time-series into (default: 10)
%
% Outputs:
% - outSet: cell array of transformed time-series, each represented as a matrix
%           where each row corresponds to a feature and columns are segment averages


function outSet = change_representation(set, segNum)
    if nargin < 2
        segNum = 10; % Default number of segments if not provided
    end

    outSet = cell(1, length(set));
    for i = 1:length(set)
        % Transpose to (time × features) for interpolation
        temp = set{i}'; 
        
        original_len = size(temp, 1);
        num_features = size(temp, 2);

        % Determine new length ensuring it is at least segNum and divisible by segNum
        newLen = max(segNum, ceil(original_len / segNum) * segNum);

        % Interpolation or repetition to adjust sequence length
        if original_len >= 3
            % Use multidimensional interpolation to smoothly resize data
            [initSize1, initSize2] = ndgrid(1:original_len, 1:num_features);
            [newSize1, newSize2] = ndgrid(linspace(1, original_len, newLen), 1:num_features);
            newData = interpn(initSize1, initSize2, temp, newSize1, newSize2);
        else
            % For very short sequences, repeat values to avoid interpolation artifacts
            newData = repmat(temp, ceil(newLen / original_len), 1);
            newData = newData(1:newLen, :);
        end

        % Calculate size of each segment (integer division)
        segSize = floor(newLen / segNum);
        representation = [];

        for d = 1:num_features
            data = newData(:, d);

            % Standardize feature data to zero mean and unit variance
            data = (data - mean(data)) / std(data);

            % Replace any NaNs, some datasets may produce them
            if any(isnan(data))
                data(isnan(data)) = 0;
            end

            % Reshape data into segments (segSize × segNum)
            % If reshape fails due to length mismatch, pad with zeros
            try
                reshaped = reshape(data, segSize, []);
            catch
                padding = zeros(segSize * segNum - length(data), 1);
                data = [data; padding];
                reshaped = reshape(data, segSize, []);
            end

            % Replace NaNs in reshaped data to avoid NaN averages
            reshaped(isnan(reshaped)) = 0;

            % Compute mean of each segment (along rows)
            segments = mean(reshaped, 1);
            segments(isnan(segments)) = 0;

            % Stack segment averages as rows (features × segments)
            representation = [representation; segments];
        end

        % Store the transformed sample in output cell array
        outSet{i} = representation;
    end
end
