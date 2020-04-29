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

mean = zeros(d,c); % store means in this matrix. Each column corresponds the mean of a class
covariance_matrices = zeros(d,d,c); % covariance matrices (i.e., covariance_matrices(:,:,c) is the covariance matrix of class c)

for i=1:c

    partial_data = x(:,y==i); % data that belong to class i
    D_c(i) = size(partial_data,2); % its size
    
    % uncomment the following line if you want to select random %75
    %partial_data(:,randperm(D_c(i))) = partial_data; % shuffle the data
    
    D_c75(i) = floor(D_c(i)*0.75); % # of samples in 75% of class c
    D_c25(i) = D_c(i) - floor(D_c(i)*0.75); % # of samples in remaining 25% of class c
    
    data75(1,i) = { partial_data( : , 1:D_c75(i) ) };
    data25(1,i) = { partial_data( : , D_c75(i)+1:end ) };
    
    mean(:,i) = (1/D_c75(i)) * sum(data75{1,i}')';
    covariance_matrices(:,:,i) = (1/D_c75(i)) * ( data75{1,i}-mean(:,i) ) * ( data75{1,i}-mean(:,i) )';
    
end

% -------Calculate the \bar{Sigma}_c matrices:------------

sigma_zero = 100000000; % sigma_0 values are taken same for each class % ------------CONTROL PARAMETER----------
sigma_zero_matrix = sigma_zero * eye(d);

bar_covariance_matrices = zeros(d,d,c);

for i = 1:c
    
    bar_covariance_matrices(:,:,i) = covariance_matrices(:,:,i)...
        + sigma_zero_matrix * inv( sigma_zero_matrix + (1/D_c75(i)) .* covariance_matrices(:,:,i))...
        * (1/D_c75(i)) * covariance_matrices(:,:,i);
end

% -------CALCULATE INVERSE AND DETERMINANT OF BAR COVARIANCE MATRICES FOR
% FUTURE USE: ---------

sigma_bar_inv = zeros(d,d,c);
sigma_bar_det = zeros(1,c);

for i = 1:c
    
    sigma_bar_inv(:,:,i) = inv(bar_covariance_matrices(:,:,i));
    sigma_bar_det(1,i) = det(bar_covariance_matrices(:,:,i));
    
end

% ------- MAKE THE CLASS PREDICTIONS: -------

predictions75 = cell(1,c); % in cell c, we will store predictions of each training sample of class c.
predictions25 = cell(1,c); % in cell c, we will store predictions of each test sample of class c.

for i=1:c
   
    pred75 = zeros( c , D_c75(i) );
    pred25 = zeros( c , D_c25(i) );
    
    for j = 1:D_c75(i)
       
        for k = 1:c
           
            pred75(k,j) = decision_function(data75{1,i}(:,j),mean(:,k),sigma_bar_det(1,k),sigma_bar_inv(:,:,k),D_c75(k));
            
        end
        
    end
    
    predictions75(1,i) = { rem(find(pred75 == min(pred75)),c) }; % in this cell, we store the predictions for each training samples
    
    for j = 1:D_c25(i)
       
        for k = 1:c
           
            pred25(k,j) = decision_function(data25{1,i}(:,j),mean(:,k),sigma_bar_det(1,k),sigma_bar_inv(:,:,k),D_c75(k));
            
        end
        
    end
    
    predictions25(1,i) = { rem(find(pred25 == min(pred25)),c) }; % in this cell, we store the predictions for each test samples
    
end

% ------- CALCULATE PERFORMANCE METRICS: -------

T_pos_training = zeros(1,c); % store TRUE POSITIVES of each class in this vector
T_pos_test = zeros(1,c);

Predicted_pos_training = zeros(1,c); % store ALL PREDICTIVE POSITIVES of each class in this vector
Predicted_pos_test = zeros(1,c);

F_neg_training = zeros(1,c); % store FALSE NEGATIVES of each class in this vector
F_neg_test = zeros(1,c);

for i = 1:c
    
   T_pos_training(1,i) = sum(rem(i,c) == predictions75{1,i});
   T_pos_test(1,i) = sum(rem(i,c) == predictions25{1,i});
    
   F_neg_training(1,i) = sum(rem(i,c) ~= predictions75{1,i});
   F_neg_test(1,i) = sum(rem(i,c) ~= predictions25{1,i});
   
   for j = 1:c
       Predicted_pos_training(1,i) = Predicted_pos_training(1,i) + sum(rem(i,c) == predictions75{1,j});
       Predicted_pos_test(1,i) = Predicted_pos_test(1,i) + sum(rem(i,c) == predictions25{1,j});
   end
   
end

PrecisionTraining = T_pos_training ./ Predicted_pos_training
PrecisionTest = T_pos_test ./ Predicted_pos_test

RecallTraining = T_pos_training ./ (T_pos_training + F_neg_training)
RecallTest = T_pos_test ./ (T_pos_test + F_neg_test)
