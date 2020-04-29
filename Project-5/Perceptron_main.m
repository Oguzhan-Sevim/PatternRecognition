close all; clc; clear all;

number_of_types = 6;
c = 8;

PCA_size = 100; % by chosing it 100, IT IS NOT APPLIED IN HO-KASHYAP

[x_75_by_type_and_classes, x_25_by_type_and_classes] = preprocess(number_of_types,PCA_size);

% Perceptron training part:
k_max = 500; % maximum iteration number in one loop
[a] = Perceptron_training(x_75_by_type_and_classes,k_max);

% Test part:
[prediction_overall75,prediction_overall25] = HoKash_test(x_75_by_type_and_classes,x_25_by_type_and_classes,a);
[predictions75,predictions25] = voting_scheme(prediction_overall75,prediction_overall25);

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