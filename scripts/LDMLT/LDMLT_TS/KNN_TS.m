function Pred_Y = KNN_TS(X_train, Y_train, X_test, M, K_vector)

n_train = size(X_train, 2);
n_test = size(X_test, 2);

% Identify unique classes in Y_train
Y_kind = Y_train(1);
n = length(Y_train);
for i = 2:n
    if sum(Y_kind == Y_train(i)) == 0
        Y_kind = [Y_train(i), Y_kind];
    end
end

for index_test = 1:n_test
    for index_train = 1:n_train
        % Compute DTW distance
        [Dist, ~, ~] = dtw_metric(X_train{index_train}, X_test{index_test}, M);
        Distance(index_train) = Dist;
    end
    
    [~, Inds] = sort(Distance, 'ascend'); % Sort distances
    
    for K_index = 1:length(K_vector)
        K = K_vector(K_index);
        counts = zeros(1, length(Y_kind));
        
        counts = zeros(1, length(Y_kind));
		for j = 1:K_index
			counts(Y_kind == Y_train(Inds(j))) = counts(Y_kind == Y_train(Inds(j))) + 1;
		end
		ids = find(counts == max(counts));
		if length(ids) == 1
			Pred_Y(K_index,index_test) = Y_kind(ids(1));
		else
			Pred_Y(K_index,index_test) = Y_train(Inds(1));
		end
    end
end