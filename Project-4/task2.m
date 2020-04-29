close all; clc; clear all;

d_prime = 50; % Select d' for PCA

data = PCA(d_prime); % call PCA algorithm for desired d'

x = data(:,2:end)'; % Extract data features as column vectors of matrix x
y = data(:,1); % Extract class information as y vector;

d = size(x,1); % d: number of features

classes = unique(y); % all class labels
c = size(classes,1); % total number of classes

% ----- PORTIONING THE DATASET BY THE CLASSES &
% CALCULATE THE MEANS AND COVARIANCE MATRICES: --------

data75 = cell(1,c); % store class separated data in these cells
data25 = cell(1,c);

D_c = zeros(1,c); % store # of samples of each class
D_c75 = zeros(1,c);
D_c25 = zeros(1,c);

for i=1:c
    
    partial_data = x(:,y==i); % data that belong to class i
    D_c(i) = size(partial_data,2); % its size
    
    % uncomment the following line if you want to select random %75
    % partial_data(:,randperm(D_c(i))) = partial_data; % shuffle the data
    
    D_c75(i) = floor(D_c(i)*0.75); % # of samples in 75% of class c
    D_c25(i) = D_c(i) - floor(D_c(i)*0.75); % # of samples in remaining 25% of class c
    
    data75(1,i) = { partial_data( : , 1:D_c75(i) ) };
    data25(1,i) = { partial_data( : , D_c75(i)+1:end ) };
    
    
end

% ------- MAKE THE CLASS PREDICTIONS on test set By K-nearest neighbor: -------

n = [16, 50]; % use 2 different number of samples
k1 = [5, 10];    % use 2 different number of neighbor size

predictions25 = cell(size(n,2),size(k1,2),c); % store the all the estimations of test set on this cell

for n_count = 1:size(n,2)
    
    for k_count = 1:size(k1,2)
        
        for i=1:c
            
            pred25 = zeros( c , D_c25(i) );
            for j = 1:D_c25(i)
                
                for k = 1:c
                    
                    pred25(k,j) = decision_functionK_nearest(data25{1,i}(:,j), data75{1,k}(:,1:n(n_count)),D_c75(k),k1(k_count));
                    
                end
                
            end
            predictions25(n_count,k_count,i) = { rem(find(pred25 == max(pred25)),c) }; % in this cell, we store the predictions for each test samples
            
        end
        
        
    end
    
end

% ------- CALCULATE PERFORMANCE METRICS: -------

T_pos_test = zeros(1,c); % store TRUE POSITIVES of each class in this vector

Predicted_pos_test = zeros(1,c); % store ALL PREDICTIVE POSITIVES of each class in this vector

F_neg_test = zeros(1,c); % store FALSE NEGATIVES of each class in this vector

n_count = 2;
k_count = 2;

for i = 1:c
    
    T_pos_test(1,i) = sum(rem(i,c) == predictions25{n_count,k_count,i});
    
    F_neg_test(1,i) = sum(rem(i,c) ~= predictions25{n_count,k_count,i});
    
    for j = 1:c
        Predicted_pos_test(1,i) = Predicted_pos_test(1,i) + sum(rem(i,c) == predictions25{n_count,k_count,j});
    end
    
end

PrecisionTest = T_pos_test ./ Predicted_pos_test

RecallTest = T_pos_test ./ (T_pos_test + F_neg_test)