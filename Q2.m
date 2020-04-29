close all; clc; clear all;

load('x_approximated.mat'); % load the new dataset
load('y.mat');

[d,n] = size(x_approximated); % d: number of features; n: number of samples

classes = unique(y); % all class labels
c = size(classes,1); % total number of classes

mean = (1/n) * sum(x_approximated')'; % overall mean

% Find and save means of each class in the class_means vector:

class_means = zeros(d,c);

for i=1:c
   
    class_means(:,i) = sum(x_approximated(:,find(y == i))')' / size(find(y == i),1);
    
end

% Calculate the projection vectors and 1D projections for all classes by one-to-rest method:

W = zeros(d,c-1);

CD = zeros(c,c); % create an empty matrix for center distances between the classes
OL = zeros(c,c); % create an empty matrix for the overlap between the classes

data_1D = zeros(c-1,n); % save the projected data on this matrix for visualization

for i=1:(c-1)
   
    sum_rest = zeros(d,1); % find the sum of all samples from classes (i+1) to c
    num_rest = 0; % find the sample number
    
    for j=(i+1):c
       
        sum_rest = sum_rest + sum(x_approximated(:,find(y == j))')';
        num_rest = num_rest + size(find(y == j),1);
        
    end
    
    
    mean_rest = sum_rest / num_rest; % mean of all classes beyond i
    
    Si = zeros(d);
    entries_i = find(y == i);
    
    for k = entries_i(1,1):entries_i(end,1)
       
        Si = Si + (x_approximated(:,k)-class_means(:,i)) * (x_approximated(:,k)-class_means(:,i))';
        
    end
    
    Srest = zeros(d); % scatter matrix for the classes between (i+1) and c
    
    for k = entries_i(end,1)+1:n
       
        Srest = Srest + (x_approximated(:,k)-mean_rest) * (x_approximated(:,k)-mean_rest)';
        
    end
    
    Sw = Si + Srest;
        
    vector = Sw \ (class_means(:,i) - mean_rest); % find the projection vector by w_i = inv(S_w)*(m_i-m)
    W(:,i) = vector / norm(vector); % make it a unit vector
    
    data_1D(i,:) = W(:,i)' * x_approximated; % find 1D projections:
      

%------------------------------------------------------
% CALCULATE THE PAIRWISE CENTER DISTANCES AND OVERLAPS:
%------------------------------------------------------

    for j = i:c
        CD(i,j) = norm( (sum(data_1D(i,find(y == i))) / size(find(y == i),1))  -  (sum(data_1D(i,find(y == j))) / size(find(y == j),1)) ); 
    end

    for j = i:c
        
        class_i = data_1D(i,find(y == i));
    	class_j = data_1D(i,find(y == j));
        
        if (min(class_i) >= max(class_j)) || (min(class_j) >= max(class_i))
            OL(i,j) = 0;
        elseif (max(class_i) >= max(class_j)) && (min(class_i) <= min(class_j))
            overlaps = size(class_j,2) + size(class_i,2) - size(find(class_i < min(class_j)),2) - size(find(class_i > max(class_j)),2);
            OL(i,j) = overlaps / (size(class_i,2) + size(class_j,2));
        elseif (max(class_j) >= max(class_i)) && (min(class_j) <= min(class_i))
            overlaps = size(class_i,2) + size(class_j,2) - size(find(class_j < min(class_i)),2) - size(find(class_j > max(class_i)),2);
            OL(i,j) = overlaps / (size(class_i,2) + size(class_j,2));
        elseif (max(class_i) >= max(class_j)) && (min(class_i) >= min(class_j))
            overlaps = size(find(class_i <= max(class_j)),2) + size(find(class_j >= max(class_i)),2);
            OL(i,j) = overlaps / (size(class_i,2) + size(class_j,2));
        elseif (max(class_j) >= max(class_i)) && (min(class_j) >= min(class_i))
            overlaps = size(find(class_j <= max(class_i)),2) + size(find(class_i >= max(class_j)),2);
            OL(i,j) = overlaps / (size(class_i,2) + size(class_j,2));
        end
    
    end

end

CD = CD + CD';

OL = OL + OL';
OL = OL - diag(diag(OL))/2;
OL = OL * 100;
OL(end,end) = 100; % :)

%------------------------------------------------------
% PLOT THE NEW 1D DATA POINTS FOR BETTER VISUALIZATION:
%------------------------------------------------------

for i =1:(c-1)
   
    data_rest = data_1D(i,:);
    last_i = find(y == i);
    data_rest(1:last_i(end)) = [];
    
    size_i = size(find(y == i),1);
    
    subplot((c-1),1,i)
    plot(data_1D(i,find(y == i)),zeros(1,size_i),'x','MarkerSize',15,'LineWidth',1);
    hold on
    plot(data_rest,zeros(1,n-last_i(end)),'o','MarkerSize',15,'LineWidth',1);
    hold off
    
    title(sprintf('Class %i vs. remaining', i));
    ax = gca;
    ax.FontSize = 18;
    
end

PairwiseCenterDistances = CD
PairwiseOverlap = OL

