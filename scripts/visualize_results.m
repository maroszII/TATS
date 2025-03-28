% Script for visualization of obtained results
% Compute mean classification accuracy for each augmentation method
meanResults = mean(results, 2);

% Extract proper method names for the bar plot
methodNames = cell(numMethods, 1);
for i = 1:numMethods
    if ischar(augmentationMethods{i}) % No augmentation case
        methodNames{i} = 'no aug';
    else
        methodName = func2str(augmentationMethods{i});
        methodNames{i} = upper(methodName(5:end)); % Extract name after "aug "
    end
end

% Plot the accuracy for each augmentation method
figure;
bar(meanResults);
xticklabels(methodNames);
xtickangle(45);
ylabel('Mean Accuracy');
title('Comparison of Augmentation Methods');
grid on;

% Compute p-values using Wilcoxon rank-sum test
numMethods = size(results, 1);
pValues = ones(numMethods); % p-wartości Matrix, 1 od diagonal

for i = 1:numMethods
    for j = i+1:numMethods
        pValues(i, j) = ranksum(results(i, :), results(j, :)); % Wilcoxon Test
        pValues(j, i) = pValues(i, j); % Symmetric matrix
    end
end

% Create a heatmap of the p-values
figure;
heatmap(methodNames, methodNames, pValues, 'Colormap', parula, 'ColorbarVisible', 'on');
xlabel('Method 1');
ylabel('Method 2');
title('Wilcoxon Test: Pairwise Comparison of Augmentation Methods');

figure;
boxplot(results', methodNames, 'LabelOrientation', 'inline');
ylabel('Accuracy');
title('Performance Distribution Across Augmentation Methods');
grid on;


baseline = mean(results(strcmp(methodNames, 'no aug'), :)); 
relativeImprovement = ((meanResults - baseline) ./ baseline) * 100;

figure;
bar(relativeImprovement);
xticklabels(methodNames);
xtickangle(45);
ylabel('Relative Improvement (%)');
title('Impact of Data Augmentation on Classification Accuracy');
grid on;

 

% MDS 2D DTW-based embeddings for augmented data
for i = 1:numMethods  % Without no aug	
    if ischar(augmentationMethods{i}) % No augmentation case, do nothing
       % Do nothing 
    else
        augFunction = augmentationMethods{i};
		visualization_of_augmented_dataset(augFunction, dataset, augSetSize); % Create plots with embeddings  
    end
end