function[a] = Perceptron_training(x_75_by_type_and_classes,k_max)

a = cell(size(x_75_by_type_and_classes,1), size(x_75_by_type_and_classes,2) );

for i = 1:size(x_75_by_type_and_classes,1)
    
    for j = 1:size(x_75_by_type_and_classes,2)
        
        data_positive = x_75_by_type_and_classes{i,j};
        data_positive = [data_positive ; ones(1,size(data_positive,2))];
        
        if j==1   
            data_negative = x_75_by_type_and_classes{i,2};
            for k = 3:size(x_75_by_type_and_classes,2)
                data_negative = [data_negative, x_75_by_type_and_classes{i,k}];
            end   
        else
            data_negative = x_75_by_type_and_classes{i,1};
            for k = 2:j-1
                data_negative = [data_negative, x_75_by_type_and_classes{i,k}];
            end
            for k = j+1:size(x_75_by_type_and_classes,2)
                data_negative = [data_negative, x_75_by_type_and_classes{i,k}];
            end
        end
        data_negative = [data_negative; ones(1,size(data_negative,2))];
        Y = [data_positive, -data_negative]';
        
        a_temp = 2 * (rand(size(Y,2),1) - .5);

        k = 1;
        while (k < k_max)
            for l = 1:size(Y,1)
                if Y(l,:)*a_temp <0
                    a_temp = a_temp + Y(l,:)';
                end
            end            
            k = k + 1;
        end
        
        a(i,j) = {a_temp};
        
    end
    
end

end