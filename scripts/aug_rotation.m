%% Augmentation method:
%% Rotation
%
% T. T. Um, F. M. J. Pfister, D. Pichler, S. Endo, M. Lang, S. Hirche,
% U. Fietzek, D. Kulić,
% Data augmentation of wearable sensor data for Parkinson’s disease monitoring using convolutional neural networks,
% ACM International Conference on Multimodal Interaction, ACM, 2017.
%
% This function augments multivariate time-series data by:
% 1. Generating random rotation angles sampled from a normal distribution,
% 2. Applying a 2D rotation to the combined time and feature axes,
% 3. Producing augmented samples that introduce variability via rotation,
%    which can improve model generalization by simulating natural sensor orientation changes.
%
% Inputs:
% - train: cell array of multivariate time-series samples (features × time)
% - trainLabels: vector of class labels corresponding to training samples
% - nDraws: number of augmented samples to generate per original sample
% - sigma: standard deviation controlling the randomness of the rotation angle (degrees, optional, default: 5.0)
%
% Outputs:
% - outTrain: cell array of augmented time-series samples
% - outTrainLabels: vector of corresponding class labels for augmented samples

function [outTrain, outTrainLabels] = aug_rotation(train, trainLabels, nDraws, sigma)
    % Set default standard deviation for rotation angle if not provided
    if nargin < 4
        sigma = 5.0;
    end

    % Preallocate output containers for efficiency
    outTrain = cell(1, length(trainLabels) * nDraws);
    outTrainLabels = zeros(1, length(trainLabels) * nDraws);
    
    % Number of original samples
    xSize = size(train, 2);
    
    counter = 1;
    for i = 1:xSize
        for iD = 1:nDraws
            temp = train{i};
            
            % Extract size: features × time_steps
            [numFeatures, numTimeSteps] = size(temp);
            
            % Generate a random rotation angle from normal distribution
            angle = sigma * randn();
            angle_rad = deg2rad(angle);
            
            % Construct the 2D rotation matrix (rotates time and one feature dimension)
            rotation_matrix = [cos(angle_rad), -sin(angle_rad); 
                               sin(angle_rad),  cos(angle_rad)];
            
            % Time indices as a row vector (0-based)
            time_indices = 0:numTimeSteps-1;
            
            % Initialize matrix for rotated series
            rotated_series = zeros(size(temp));
            
            % Apply rotation per time step:
            % For each time step, form a 2D point from time and first feature,
            % then rotate it and use only the rotated feature coordinate as new value.
            % This simulates a rotation effect on the signal progression.
            for t = 1:numTimeSteps
                point = [time_indices(t); temp(:, t)];
                
                % Rotation is applied only to the first two dimensions:
                % time and the first feature channel (assuming at least one feature).
                rotated_point = rotation_matrix * point(1:2);
                
                % Store only the rotated feature value (second coordinate) at each time step
                rotated_series(:, t) = rotated_point(2);
            end            
           
            outTrain{counter} = rotated_series;
            outTrainLabels(counter) = trainLabels(i);
            
            counter = counter + 1;
        end
    end
end
