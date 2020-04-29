function[x_75_by_type_and_classes, x_25_by_type_and_classes] = preprocess(number_of_types,PCA_size)

%data = textread('Test.txt'); % FOR TEST DATA

data = textread('Dataset8Class.txt'); % read data from text file
data = data(:,1:end-1); % get rid of the last column which is full of zeros
data = sortrows(data,1); % sort the data set by the classes (first row)
x = data(:,2:end)'; % Extract data features as column vectors of matrix x
y = data(:,1); % Extract class information as y vector

x = standardization(x); % apply normalization

[d,n] = size(x); % d: number of features; n: number of samples

classes = unique(y); % all class labels
c = size(classes,1); % total number of classes

D_c = zeros(1,c); % store # of samples of each class
D_c75 = zeros(1,c);
D_c25 = zeros(1,c);

x_75 = cell(1,c); % store test and training data in different cells
x_25 = cell(1,c);

for i=1:c
    partial_data = x(:,y==i); % data that belong to class i
    D_c(i) = size(partial_data,2); % its size
    
    % uncomment the following line if you want to select random %75
    partial_data(:,randperm(D_c(i))) = partial_data; % shuffle the data
    
    D_c75(i) = floor(D_c(i)*0.75); % # of samples in 75% of class c
    D_c25(i) = D_c(i) - floor(D_c(i)*0.75); % # of samples in remaining 25% of class c
    
    x_75(1,i) = {partial_data( : , 1:D_c75(i) )};
    x_25(1,i) = {partial_data( : , D_c75(i)+1 : end )};
    
end

x_75_by_type_and_classes = cell(number_of_types,c); % training data is separated in different cells by their class and feature type info.
x_25_by_type_and_classes = cell(number_of_types,c); % same for test data

for i = 1:number_of_types
    
    for j = 1:c
        
        x_75_by_type_and_classes(i,j) = { x_75{1,j}( (i-1)*d/number_of_types+1 : i*d/number_of_types , : ) };
        x_25_by_type_and_classes(i,j) = { x_25{1,j}( (i-1)*d/number_of_types+1 : i*d/number_of_types , : ) };
        
    end
    
end

if PCA_size ~= 100 % PCA will be applied if ne new size is chosen different than 100
    [x_75_by_type_and_classes,x_25_by_type_and_classes] = PCA(x_75_by_type_and_classes, x_25_by_type_and_classes, PCA_size);
end

end