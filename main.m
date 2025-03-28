% -------------------------------------------------------------------------
% TATS: Toolbox for Augmenting Time Series
% Demo 
% -------------------------------------------------------------------------
% Author: Dawid Warchoł and Mariusz Oszust
% Affiliation: Rzeszow University of Technology
% Email: dawwar@prz.edu.pl, marosz@kia.prz.edu.pl 
% Date: 2025-03-28
% Version: 1.0  
% -------------------------------------------------------------------------
clc, clear all, close all;
% Add paths to folders						
addpath(genpath(fullfile(pwd, 'datasets')));
addpath(genpath(fullfile(pwd, 'scripts')));

disp('TATS: Toolbox for Augmenting Time Series')


% Available datasets:
    % 'FLORENCE' - https://www.micc.unifi.it/resources/datasets/florence-3d-actions-dataset/
    % 'KARD' - https://data.mendeley.com/datasets/k28dtm7tr6/1
    % 'MSRA' - https://sites.google.com/view/wanqingli/data-sets/msr-action3d  
    % 'UTD' - https://personal.utdallas.edu/~kehtar/UTD-MHAD.html
    % 'UTK' - http://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html
    % 'VISAPP' - https://doi.org/10.5220/0004217606200625
    % 'AReM' - https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem
    % 'AUSLAN' - https://archive.ics.uci.edu/dataset/115/australian+sign+language+signs+high+quality
    % 'ECG' - https://www.cs.cmu.edu/~bobski/data/data.html
    % 'EEG' - https://archive.ics.uci.edu/dataset/121/eeg+database
    % 'GesturePhaseDetect' - https://archive.ics.uci.edu/dataset/302/gesture+phase+segmentation
    % 'KickVsPunch' - http://mocap.cs.cmu.edu
    % 'LIBRAS' - https://archive.ics.uci.edu/dataset/181/libras+movement
    % 'MovementAAL' - https://archive.ics.uci.edu/dataset/348/indoor+user+movement+prediction+from+rss+data
    % 'OccupancyDetect' - https://archive.ics.uci.edu/dataset/357/occupancy+detection
    % 'Ozone' - https://archive.ics.uci.edu/dataset/172/ozone+level+detection
    % 'Pendigits' - https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits (training and testing sets swapped for higher difficulty)
dataset = 'VISAPP'; 

% Size of augmented set (that will be added to the original training set)
% 1 means augmented set is 100% of the original training set size (so the training set is double the original size)
augSetSize = 1;

% Number of test repetitions
% It will be ignored if 'no aug' is set
repetitions = 10;

% Available classifiers:
% 'DTW' - k-Nearest Neighbors + Dynamic Time Warping
% 'LSTM' - Long Short-Term Memory Network
classifier = 'LSTM';

% Classifier parameters
if strcmpi(classifier,'DTW')
    % k-NN + DTW
    parameters.k = 1;
    parameters.windowSize = 5;
    parameters.metric = 'euclidean'; % Available metrics: 'euclidean', 'absolute', 'squared'
elseif strcmpi(classifier,'LSTM')
    % LSTM
    parameters.maxEpochs = 125;
    parameters.miniBatchSize = 16;
    parameters.initialLearnRate = 0.0008;
    parameters.gradientThreshold = 1;
    parameters.bidirectional = true;
    parameters.processingUnit = 'gpu' % Available options: 'cpu', 'gpu' (requires NVIDIA card and CUDA drivers)
else
    error('Uknown classifier name.');
end

%%% ------------- Single augmentation method  
% Available augmentation methods:
% @aug_ws - Window Slicing
% @aug_ww - Window Warping
% @aug_mw - Magnitude Warping
% @aug_dba - DTW Barycenter Averaging
% @aug_spawner - Suboptimal Warped Time Series Generator
% @aug_arspawner - Action Recognition SPAWNER 
% @aug_eww - Extended Window Warping
% @aug_adder - Adder
% 'no aug' - no augmentation
augmentationMethod = @aug_ws;

%% Running one experiment with augmentation to obtain classification accuracy
%accuracy = validation_tests(augmentationMethod, classifier, dataset, repetitions, parameters, augSetSize); % comment / uncomment if needed

%%% ------------- Multiple augmentation methods and visualization 

% Turn multiprocessing on or off.
% When turned on, total testing time will be greatly reduced (especially on a CPU with many cores).
% However, much less information will be displayed on the console (less progress and result logs).
% Note that multiprocessing will not be used if LSTM with GPU is chosen as a classifier.
% --Requires Parallel Computing Toolbox--
multiprocessing = true;

% Select multiple augmentation methods
augmentationMethods = {@aug_ws, @aug_ww, @aug_mw, @aug_dba, @aug_spawner, @aug_arspawner, ...
                       @aug_eww, @aug_adder, 'no aug'}; 

run_multiple_augmentation_methods  % comment / uncomment if needed
visualize_results                  % comment / uncomment if needed