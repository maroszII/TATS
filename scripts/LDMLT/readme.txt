The fold file contains:
- readme.txt			This file
- demo_LDMLT_TS.m		A demo to test the performance of our algorithm.


- LDMLT Folder			This fold contains the files of LDMLT_TS algorithm.
--LDMLT_TS.m			Generate Mahalanobis matrix according to LDMLT_TS algorithm.
--Select_Triplets.m		Generate triplets according to dynamic triplets building algorithm
--Order_Check.m			Compute the Mahalanobis distance of all the sample pairs and the disorder using the the current Mahalanobis matrix M
--SetDefaultParams.m	Sets default parameters
- KNN_TS.m 				Perform the k-nearest neighbors (KNN) classification algorithm.
- dtw_metric.m 			Compare two Multivariate Time Series using Mahalanobis Distance based Dynamic Time Warping Measure.


- data Folder
--JapaneseVowels_TRAIN_X.mat JapaneseVowels_TRAIN_Y.mat JapaneseVowels_TEST_X.mat JapaneseVowels_TEST_Y.mat

The data is used to test the performance of LDMLT_TS.    Please refer "https://archive.ics.uci.edu/ml/databases/JapaneseVowels/" for details.		                       


Reference:
[1] Jiangyuan Mei, Meizhu Liu, Hamid Reza Karimi, and Huijun Gao, "LogDet Divergence based Metric Learning with Triplet Constraints and Its Applications", IEEE Transactions on image processing, Accepted.
[2] Jiangyuan Mei, Meizhu Liu, Yuan-Fang Wang, and Huijun Gao, "Learning a Mahalanobis Distance based Dynamic Time Warping Measure for Multivariate Time Series Classification".

