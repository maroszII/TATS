%% Visualization and Statistical Analysis of Classification Results
%
% This script computes and visualizes the mean classification accuracy
% obtained from multiple augmentation methods. It performs statistical
% comparison using the Wilcoxon rank-sum test and visualizes:
% - mean accuracy bar plots,
% - p-value heatmap for pairwise method comparisons,
% - box plots of accuracy distributions,
% - relative improvement over baseline (no augmentation),
% - and generates 2D MDS embeddings for augmented datasets.
%
% Inputs assumed to be available in the workspace:
% - accuracies: matrix of accuracies (methods × repetitions)
% - augmentationMethods: cell array of augmentation method handles or 'no aug'
% - dataset: dataset name string (for visualization)
% - parameters.augSetSize: augmentation set size multiplier

%% Compute mean accuracy for each augmentation method
meanAccuracies = mean(accuracies, 2);

%% Extract readable method names for plotting
numMethods = size(accuracies, 1);
methodNames = cell(numMethods, 1);
for i = 1:numMethods
    if ischar(augmentationMethods{i}) % Handle 'no augmentation' case
        methodNames{i} = 'no aug';
    else
        methodName = func2str(augmentationMethods{i});
        methodNames{i} = upper(methodName(5:end)); % Remove 'aug_' prefix and capitalize
    end
end

%% Plot mean accuracy for each augmentation method
figure;
bar(meanAccuracies);
xticklabels(methodNames);
xtickangle(45);
ylabel('Mean Accuracy');
title('Comparison of Augmentation Methods');
grid on;

%% Plot mean execution time for each augmentation method
% Ensure methodNames is a column
methodNames = methodNames(:);
% Use provided mean times
meanTimes = augTimes';
% Ensure orientation consistency
if numel(meanTimes) ~= numel(methodNames)
    meanTimes = meanTimes(:);
end
% Logical index for 'no aug', if is added
idxNoAug = strcmpi(methodNames, 'no aug');
% Remove 'no aug' if it exists
meanTimesFiltered = meanTimes(~idxNoAug);
methodNamesFiltered = methodNames(~idxNoAug);
% Plot with logarithmic Y-axis as some methods may compute much longer than others
figure;
bar(meanTimesFiltered);
set(gca, 'YScale', 'log');
xticks(1:numel(methodNamesFiltered));
xticklabels(methodNamesFiltered);
xtickangle(45);
ylabel('Mean Execution Time (log scale) [s]');
title('Comparison of Execution Time of Augmentation Methods');
grid on;



%% Compute pairwise p-values using Wilcoxon rank-sum test
pValues = ones(numMethods); % Initialize symmetric matrix of p-values
for i = 1:numMethods
    for j = i+1:numMethods
        pValues(i, j) = ranksum(accuracies(i, :), accuracies(j, :));
        pValues(j, i) = pValues(i, j);
    end
end

%% Plot heatmap of p-values from statistical tests
figure;
heatmap(methodNames, methodNames, pValues, 'Colormap', parula, 'ColorbarVisible', 'on');
xlabel('Method 1');
ylabel('Method 2');
title('Wilcoxon Test: Pairwise Comparison of Augmentation Methods');

% Create box plots showing accuracy distributions per method
figure;
boxplot(accuracies', methodNames, 'LabelOrientation', 'inline');
ylabel('Accuracy');
title('Performance Distribution Across Augmentation Methods');
grid on;

%% Calculate and plot relative improvement over baseline (no augmentation)
baseline = mean(accuracies(strcmp(methodNames, 'no aug'), :)); 
relativeImprovement = ((meanAccuracies - baseline) ./ baseline) * 100;

figure;
bar(relativeImprovement);
xticklabels(methodNames);
xtickangle(45);
ylabel('Relative Improvement (%)');
title('Impact of Data Augmentation on Classification Accuracy');
grid on;


%% Generate 2D MDS embeddings visualizations for augmented datasets
for i = 1:numMethods
    if ischar(augmentationMethods{i}) % Skip 'no augmentation'
        continue;
    else
        augFunction = augmentationMethods{i};
        visualization_of_augmented_dataset(augFunction, dataset, parameters.augSetSize);
    end
end
