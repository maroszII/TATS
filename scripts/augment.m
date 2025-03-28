function [OutTrain, OutTrainLabels] = augment(train,trainlabels,augFunction,multiplicity)

    [OutTrain, OutTrainLabels] = augFunction(train,trainlabels,multiplicity);
   
end