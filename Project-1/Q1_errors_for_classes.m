close all; clc; clear all;

data = textread('DataSet.txt'); % read data from text file
data = data(:,1:end-1); % get rid of the last column which is full of zeros
data = sortrows(data,1); % sort the data set by the classes (first row)

x = data(:,2:end)'; % Extract data features as column vectors of matrix x
y = data(:,1); % Extract class information as y vector;

[d,n] = size(x); % d: number of features; n: number of samples

classes = unique(y); % all class labels
c = size(classes,1); % total number of classes

mean = (1/n) * sum(x')'; % sample mean

S = zeros(d); % create an empty scatter matrix

for i = 1:n
    S = S + (x(:,i)-mean)*(x(:,i)-mean)'; % add on scatter matrix by each sample
end

[V,lambda] = eig(S); % calculate the normalized eigenvectors and eigenvalues of S matrix
%lambda is the diagonal matrix which contains the eigenvalues of S. Also, V
%contains the correponding eigenvectors
[diag,ind] = sort(diag(lambda));% Sort the elements of lambda and V in an ascending order
lambda = lambda(ind,ind);
V = V(:,ind);

A = zeros(n,d); % this matrix contains the a_ki coefficients

for i = 1:n
    for j = 1:d
       
        A(i,j) = V(:,end+1-j)' * (x(:,i) - mean); % start from the eigenvector with the largest eigenvalue
        
    end
end

x_approximated = zeros(d,n);

error = zeros(1,n); % store the errors each sample
d_prime = 50; % Choose d'=50

for k = 1:n
        
    projection = zeros(d,1);
       
    for i = 1:d_prime
           
        projection = projection + A(k,i)*V(:,end+1-i);
            
    end
        
    x_approximated(:,k) = mean + projection;
    error(k) = norm(x_approximated(:,k) - x(:,k) )^2; % error of kth sample
        
end

average_errors_in_each_class = zeros(1,c); % save the average errors of each class in this vector

for i=1:c
   
    average_errors_in_each_class(i) = sum(error(find(y == i))) / size(find(y == i),1); % average error of class i
    
end

average_errors_in_each_class

%-------------------------------------------------------------------
% THE REMAINING PART SAVES THE NEW DATASET FOR THE FUTURE USE IN Q2: 
%-------------------------------------------------------------------

save('x_approximated.mat','x_approximated');
save('y.mat','y');
