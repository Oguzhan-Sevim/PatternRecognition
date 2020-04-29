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

error = zeros(1,d); % store the errors for different d' values

for d_prime = 1:d % perform PCA for all possible d' values from 1 to d
    
    d_prime

    for k = 1:n
        
        projection = zeros(d,1);
       
        for i = 1:d_prime
           
            projection = projection + A(k,i)*V(:,end+1-i);
            
        end
        
        error(d_prime) = error(d_prime) + norm(mean - x(:,k) + projection)^2;
        
    end

end

plot(log10(error),'LineWidth',5);
ylabel('log_{10}J(d_{prime})');
xlabel('d_{prime}');