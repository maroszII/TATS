%% Script for augmenting dataset using the specified augmentation method
%
% This function acts as a wrapper that applies a specified augmentation
% method to the training dataset.
%
% Inputs:
% - train: cell array of original multivariate time-series samples
% - trainlabels: vector of class labels corresponding to the training samples
% - augFunction: function handle to the augmentation method, or string 'no aug'
% - multiplicity: number of augmentation iterations to apply per sample
%
% Outputs:
% - OutTrain: cell array containing augmented training samples
% - OutTrainLabels: vector containing labels for the augmented samples

function [OutTrain, OutTrainLabels] = augment(train, trainlabels, augFunction, multiplicity)
    % If no augmentation requested, simply return original data
    if ischar(augFunction) && strcmpi(augFunction, 'no aug')
        OutTrain = train;
        OutTrainLabels = trainlabels;
        return
    end
    
    % Otherwise, apply the specified augmentation function
    [OutTrain, OutTrainLabels] = augFunction(train, trainlabels, multiplicity);
end
