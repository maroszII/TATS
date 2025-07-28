%% Augmentation method:
%% SPAWNER - Suboptimal Warped Time Series Generator
%
% K. Kamycki, T. Kapuściński, M. Oszust,
% Data augmentation with suboptimal warping for time-series classification,
% Sensors 20 (1) (2020).
%
% This function augments multivariate time-series data by:
% 1. Computing pairwise DTW distances with constrained windows,
% 2. Identifying pairs of samples from the same class with low distance,
% 3. Generating new samples by partially warping and blending subsegments of these pairs,
% 4. Adding Gaussian noise around the average warped segments to create realistic variability.
%
% Inputs:
% - train: cell array of multivariate time-series samples (features × time)
% - trainLabels: vector of class labels corresponding to training samples
% - nDraws: number of augmented samples to generate per matching pair
%
% Outputs:
% - outTrain: cell array with augmented time-series samples
% - outTrainLabels: corresponding labels for augmented data

function [outTrain, outTrainLabels] = aug_spawner(train, trainLabels, nDraws)
    % Initialize output containers sized for maximum possible augmented samples
    outTrain = cell(1, length(trainLabels) * nDraws);
    outTrainLabels = zeros(1, length(trainLabels) * nDraws);
    xSize = size(train, 2);

    % Preallocate matrix to store pairwise normalized DTW distances
    arrPod = zeros([xSize, xSize]);

    % Calculate pairwise DTW distances with window constraints for efficiency
    for i = 1:xSize  
        tempI = train{i};
        for j = i+1:xSize  
            tempJ = train{j};

            % Define Sakoe-Chiba window size as 10% of longer sequence length
            window = ceil(max(size(tempI,2), size(tempJ,2)) / 10);

            try 
                % Compute DTW distance normalized by total sequence length
                dt = dtw(tempI, tempJ, window) / (size(tempI,2) + size(tempJ,2));
                if isnan(dt)
                    % Fall back to full DTW if constrained DTW fails
                    dt = dtw(tempI, tempJ) / (size(tempI,2) + size(tempJ,2));
                end
            catch                
                dt = 1e10;
            end

            arrPod(i,j) = dt; 
            arrPod(j,i) = dt;
        end
    end

    % Calculate average intra-class DTW distances for each sample (optional, for analysis)
    arrAvgClass = zeros(xSize,1);
    for i = 1:xSize
        sumDist = 0;
        count = 0;
        for j = 1:xSize
            if trainLabels(i) == trainLabels(j)
                sumDist = sumDist + arrPod(i,j);
                count = count + 1;
            end
        end
        % Exclude self-distance by subtracting one from count
        arrAvgClass(i) = sumDist / max(count - 1,1);
    end

    % Calculate minimal intra-class DTW distances (optional, for analysis)
    arrAvgClassMin = zeros(xSize,1);
    for i = 1:xSize
        minimum = 1e9; % Large initial value
        for j = 1:xSize
            if trainLabels(i) == trainLabels(j) && i ~= j
                if arrPod(i,j) < minimum
                    minimum = arrPod(i,j);
                end
            end
        end
        arrAvgClassMin(i) = minimum;
    end

    % Calculate average DTW distance per class (optional, for analysis)
    arrClass = zeros(max(trainLabels),1);
    for n = 1:max(trainLabels)
        sumDist = 0;
        count = 0;
        for i = 1:xSize
            for j = i+1:xSize
                if trainLabels(j) == n
                    sumDist = sumDist + arrPod(i,j);
                    count = count + 1;
                end
            end
        end
        arrClass(n) = sumDist / max(count - 1,1);
    end

    % Initialize tracking of already processed pairs to avoid duplicates
    checked = zeros([xSize, xSize]);
    counter = 1;

    % Main loop to generate augmented samples
    for i = 1:xSize
        tempI = train{i};
        odl4I = arrPod(i,:);

        % Sort indices by ascending DTW distance to sample i
        [~, ind] = sort(odl4I, 'ascend');
        % Remove self index (first element)
        ind = ind(2:end);

        % Iterate over sorted neighbors to find candidates for augmentation
        for j = ind
            tempJ = train{j};
            % Only process pairs from the same class
            if trainLabels(i) == trainLabels(j)
                % Check if pair (i,j) not yet augmented
                if checked(i,j) <= 0
                    % Mark pair as processed (bidirectional)
                    checked(i,j) = checked(i,j) + 1;
                    checked(j,i) = checked(j,i) + 1;

                    % Generate nDraws augmented samples per pair
                    for draw = 1:nDraws
                        % Random cut point proportions within each sequence
                        randAB = rand();
                        drawA = ceil(size(tempI, 2) * randAB);
                        drawB = ceil(size(tempJ, 2) * randAB);

                        % Split sequences at cut points into two parts
                        t1a = tempI(:, 1:drawA); 
                        t1b = tempI(:, drawA+1:end);     
                        t2a = tempJ(:, 1:drawB); 
                        t2b = tempJ(:, drawB+1:end); 

                        % Define window sizes for DTW sub-alignments on each segment
                        ct1a = ceil(size(t1a, 2) / 10);
                        ct2a = ceil(size(t2a, 2) / 10);
                        ct1b = ceil(size(t1b, 2) / 10);
                        ct2b = ceil(size(t2b, 2) / 10);

                        % Choose maximum window for first segments, with minimum 1
                        window = max(ct1a, ct2a);
                        if window < 1, window = 1; end

                        try     
                            % Compute DTW alignment paths for first segments
                            [~, ix1, iy1] = dtw(t1a, t2a, window);
                        catch
                            % Skip pair if DTW fails
                            continue
                        end

                        % Extract aligned subsequences or original if too short
                        if size(t1a, 2) > 1 || size(t2a, 2) > 1
                            onewarp1 = t1a(:, ix1);
                            twowarp1 = t2a(:, iy1);
                        else
                            onewarp1 = t1a;
                            twowarp1 = t2a;
                        end

                        % Repeat for second segments
                        window = max(ct1b, ct2b);
                        if window < 1, window = 1; end

                        [~, ix2, iy2] = dtw(t1b, t2b, window);
                        if size(t1b, 2) > 1 || size(t2b, 2) > 1
                            onewarp2 = t1b(:, ix2);
                            twowarp2 = t2b(:, iy2);
                        else
                            onewarp2 = t1b;
                            twowarp2 = t2b;
                        end

                        % Generate synthetic segments by averaging aligned pairs plus Gaussian noise
                        together1 = normrnd(0.5 * (onewarp1 + twowarp1), 0.05 * abs(onewarp1 - twowarp1));
                        together2 = normrnd(0.5 * (onewarp2 + twowarp2), 0.05 * abs(onewarp2 - twowarp2));

                        % Concatenate augmented segments  
                        together = [together1, together2];

                        % Store augmented sample and label
                        outTrainLabels(counter) = trainLabels(i);
                        outTrain{counter} = together;
                        counter = counter + 1;
                    end
                end
            end
        end
    end

    % Trim unused preallocated cells  
    outTrain(counter:end) = [];
    outTrainLabels(counter:end) = [];
end
