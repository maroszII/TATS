%% TATS: Toolbox for Augmenting Time-Series
% Demo script for time-series data augmentation and classification.
%
% Authors: Dawid Warchoł & Mariusz Oszust
%
% Affiliation: Rzeszow University of Technology
%
% Emails: dawwar@prz.edu.pl, marosz@kia.prz.edu.pl  
%
% Date: 2025-07-28 
% Version: 2.0  

%% Initialization
% Clear console, workspace, and close all figures.
clc;
clear all;
close all;

% Add paths to folders with datasets and scripts.
addpath(genpath(fullfile(pwd, 'datasets')));
addpath(genpath(fullfile(pwd, 'scripts')));

disp('TATS: Toolbox for Augmenting Time-Series')

%% Available Datasets
% The following datasets are available for experimentation:
%
% * 'FLORENCE' - https://www.micc.unifi.it/resources/datasets/florence-3d-actions-dataset/  
% * 'KARD' - https://data.mendeley.com/datasets/k28dtm7tr6/1  
% * 'MSRA' - https://sites.google.com/view/wanqingli/data-sets/msr-action3d  
% * 'UTD' - https://personal.utdallas.edu/~kehtar/UTD-MHAD.html  
% * 'UTK' - http://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html  
% * 'VISAPP' - https://www.scitepress.org/Link.aspx?doi=10.5220/0004217606200625, https://web.archive.org/web/20121025131124/https://mll.sehir.edu.tr/visapp2013
% * 'AReM' - https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem  
% * 'AUSLAN' - https://archive.ics.uci.edu/dataset/115/australian+sign+language+signs+high+quality  
% * 'ECG' - https://www.cs.cmu.edu/~bobski/data/data.html  
% * 'EEG' - https://archive.ics.uci.edu/dataset/121/eeg+database  
% * 'GesturePhaseDetect' - https://archive.ics.uci.edu/dataset/302/gesture+phase+segmentation  
% * 'KickVsPunch' - http://mocap.cs.cmu.edu, https://zenodo.org/records/10852865, http://link.springer.com/article/10.1007/s10618-015-0425-y  
% * 'LIBRAS' - https://archive.ics.uci.edu/dataset/181/libras+movement  
% * 'MovementAAL' - https://archive.ics.uci.edu/dataset/348/indoor+user+movement+prediction+from+rss+data  
% * 'OccupancyDetect' - https://archive.ics.uci.edu/dataset/357/occupancy+detection  
% * 'Ozone' - https://archive.ics.uci.edu/dataset/172/ozone+level+detection  
% * 'Pendigits' - https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits  
%   (training and testing sets swapped for higher difficulty)"

dataset = 'UTK'; 

%% Augmentation and Repetition Parameters
% Define augmentation set size and the number of test repetitions.
augSetSize = 1;           % 1 means 100% of original training set size.
repetitions = 10;         % Number of test repetitions (ignored if 'no aug' is selected).

%% Classifier Selection
% Choose from the following classifiers:
%
% * 'DTW' - k-Nearest Neighbors + Dynamic Time Warping  
% * 'LDMLT' - LogDet Divergence-Based Metric Learning with Triplet Constraints  
% * 'DE' - Delay Embedding  
% * 'LSTM' - Long Short-Term Memory Network  
% * 'GRU' - Gated Recurrent Unit Network  
% * 'Transformer' - Transformer Network  

classifier = 'LSTM';

% Set classifier-specific parameters.
if strcmpi(classifier,'DTW')
    parameters.k = 1;
    parameters.windowSize = 5;
    parameters.metric = 'euclidean';
elseif strcmpi(classifier,'LDMLT')
	parameters.k = 1;
	parameters.tripletsfactor = 20;
	parameters.epochs = 15;
	parameters.alphafactor = 5;
elseif strcmpi(classifier,'DE')
	parameters.step = 5;
	parameters.dim = 2;
	parameters.slide = 2;
	parameters.gridSize = 2/20;
	parameters.isGrid = false;
	parameters.alpha = 2;
	parameters.beta = 2;
	parameters.filter = 0.5;
elseif strcmpi(classifier,'LSTM')
    parameters.maxEpochs = 125;
    parameters.miniBatchSize = 16;
    parameters.initialLearnRate = 0.0008;
    parameters.gradientThreshold = 1;
    parameters.bidirectional = true;
    parameters.processingUnit = 'gpu';
elseif strcmpi(classifier,'GRU')
    parameters.maxEpochs = 125;
    parameters.miniBatchSize = 16;
    parameters.initialLearnRate = 0.001;
    parameters.gradientThreshold = 1;
    parameters.processingUnit = 'gpu';
elseif strcmpi(classifier,'Transformer')
    parameters.maxEpochs = 125;
    parameters.miniBatchSize = 16;
    parameters.initialLearnRate = 0.0008;
    parameters.gradientThreshold = 1;
    parameters.processingUnit = 'gpu';
    parameters.dropout = 0.1;
    parameters.numHeads = 4;
else
    error('Unknown classifier name.');
end

%% Single Augmentation Method
% Available methods:
%
% * @aug_ws - Window Slicing  
% * @aug_ww - Window Warping  
% * @aug_mw - Magnitude Warping  
% * @aug_jitter - Jittering  
% * @aug_rotation - Rotation  
% * @aug_dba - DTW Barycenter Averaging  
% * @aug_spawner - Suboptimal Warped Time Series Generator  
% * @aug_arspawner - Action Recognition SPAWNER  
% * @aug_eww - Extended Window Warping  
% * @aug_adder - Adder  
% * 'no aug' - No augmentation

augmentationMethod = @aug_adder;

% Run one experiment using the selected augmentation method.
% accuracy = validation_tests(augmentationMethod, classifier, dataset, repetitions, parameters, augSetSize);

%% Multiple Augmentation Methods (Optional)
% Evaluate multiple augmentation methods and visualize results.
%
% Multiprocessing greatly reduces testing time on multi-core CPUs.  
% It is not supported when using GPU-based deep learning classifier (LSTM, GRU or Transformer).
multiprocessing = false; 

augmentationMethods = {
    @aug_ws, @aug_ww, @aug_mw, @aug_jitter, @aug_rotation, @aug_dba, ...
    @aug_spawner, @aug_arspawner, @aug_eww, @aug_adder, 'no aug' }; 

% Uncomment below lines to run batch tests and visualize results.
run_multiple_augmentation_methods  
visualize_results  

