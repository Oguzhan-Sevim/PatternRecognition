function[x_75_after_PCA,x_25_after_PCA] = PCA(x_75_by_type_and_classes, x_25_by_type_and_classes, d_prime)

x_75_after_PCA = cell(size(x_75_by_type_and_classes));
x_25_after_PCA = cell(size(x_25_by_type_and_classes));

% apply PCA to training set:
for i_count = 1:size(x_75_by_type_and_classes,1)
    
    for j_count = 1:size(x_75_by_type_and_classes,2)

        x = x_75_by_type_and_classes{i_count,j_count};
        
        [d,n] = size(x); % d: number of features; n: number of samples
        
        mean = (1/n) * sum(x')'; % sample mean
        
        S = zeros(d); % create an empty scatter matrix
        
        for i = 1:n
            S = S + (x(:,i)-mean)*(x(:,i)-mean)'; % add on scatter matrix by each sample
        end
        
        [V,lambda] = eig(S); % calculate the normalized eigenvectors and eigenvalues of S matrix
        %lambda is the diagonal matrix which contains the eigenvalues of S. Also, V
        %contains the correponding eigenvectors
        [~,ind] = sort(diag(lambda));% Sort the elements of lambda and V in an ascending order
        lambda = lambda(ind,ind);
        V = V(:,ind);
        
        A = zeros(n,d); % this matrix contains the a_ki coefficients
        
        for i = 1:n
            for j = 1:d
                
                A(i,j) = V(:,end+1-j)' * (x(:,i) - mean); % start from the eigenvector with the largest eigenvalue
                
            end
        end
        
        x_75_after_PCA(i_count,j_count) = {A(:,1:d_prime)'};
        
    end
    
end



% apply PCA to test set:
for i_count = 1:size(x_75_by_type_and_classes,1)
    
    for j_count = 1:size(x_25_by_type_and_classes,2)
        
        x = x_25_by_type_and_classes{i_count,j_count};
        
        [d,n] = size(x); % d: number of features; n: number of samples
        
        mean = (1/n) * sum(x')'; % sample mean
        
        S = zeros(d); % create an empty scatter matrix
        
        for i = 1:n
            S = S + (x(:,i)-mean)*(x(:,i)-mean)'; % add on scatter matrix by each sample
        end
        
        [V,lambda] = eig(S); % calculate the normalized eigenvectors and eigenvalues of S matrix
        %lambda is the diagonal matrix which contains the eigenvalues of S. Also, V
        %contains the correponding eigenvectors
        [~,ind] = sort(diag(lambda));% Sort the elements of lambda and V in an ascending order
        lambda = lambda(ind,ind);
        V = V(:,ind);
        
        A = zeros(n,d); % this matrix contains the a_ki coefficients
        
        for i = 1:n
            for j = 1:d
                
                A(i,j) = V(:,end+1-j)' * (x(:,i) - mean); % start from the eigenvector with the largest eigenvalue
                
            end
        end
        
        x_25_after_PCA(i_count,j_count) = {A(:,1:d_prime)'};
        
    end
    
end


end