% calculateMetrics Calculates multiclass classification metrics.
%   [accuracy, metrics] = calculate_metrics(groundTruth, predictedLabels) computes Precision,
%   Recall, F1-Score, and Accuracy based on ground truth and predicted
%   labels (categorical arrays).
%
%   Inputs:
%   - groundTruth: 1D categorical array of actual labels (TEST_Y)
%   - predictedLabels: 1D categorical array of model predictions
%
%   Output:
%	- accuracy: global accuracy
%   - metrics: A MATLAB Table containing per-class metrics
%              and a summary row with macro-averages

function [accuracy, metrics] = calculate_metrics(groundTruth, predictedLabels)
    % Ensure inputs are column vectors to avoid dimension mismatch
    if isrow(groundTruth); groundTruth = groundTruth'; end
    if isrow(predictedLabels); predictedLabels = predictedLabels'; end

    % Compute Confusion Matrix
    % 'C' is the matrix, 'order' contains the class names
    [C, order] = confusionmat(groundTruth, predictedLabels);

    % Calculate components for each class (One-vs-Rest approach)
    TP = diag(C);               % True Positives
    FP = sum(C, 1)' - TP;       % False Positives (Column sum - TP)
    FN = sum(C, 2) - TP;        % False Negatives (Row sum - TP)
    totalSamples = sum(C(:));   % Total number of observations

    % Calculate metrics per class
    precision = TP ./ (TP + FP);
    recall = TP ./ (TP + FN);   % This is Sensitivity
    F1_score = 2 * (precision .* recall) ./ (precision + recall);

    % Handle NaN values (e.g., division by zero if a class is missing)
    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    F1_score(isnan(F1_score)) = 0;

    % Calculate Aggregated Metrics (Macro and Global)
    macroPrecision = mean(precision);
    balancedAccuracy = mean(recall); % This is Macro Recall
    macroF1 = mean(F1_score);
    accuracy = sum(TP) / totalSamples;

    % Construct the Output Table
    % Convert class names to cell array of strings
    classNames = cellstr(order); 
    
    % Append the summary label
    classNames{end+1, 1} = 'Average / Global';

    % Prepare data vectors with the summary row appended at the end
    dataPrecision = [precision; macroPrecision];
    dataRecall = [recall; balancedAccuracy];
    dataF1 = [F1_score; macroF1];

    % Create the output metric table
    metrics = table(classNames, dataPrecision, dataRecall, dataF1, ...
        'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score'});
end