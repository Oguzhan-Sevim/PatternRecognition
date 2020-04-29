function[a,b,error_save] = HoKash_training(x_75_by_type_and_classes,eta)

a = cell(size(x_75_by_type_and_classes,1), size(x_75_by_type_and_classes,2) );
b = cell(size(x_75_by_type_and_classes,1), size(x_75_by_type_and_classes,2) );

error_save = cell(size(x_75_by_type_and_classes,1), size(x_75_by_type_and_classes,2) );

k_max = 200; % maximum iteration number in one loop

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
        b_temp = rand(size(Y,1),1);
        
        b_min = ones(size(Y,1),1) * 10^(-3);
        
        error_save(i,j) = {zeros(1,k_max)};
        
        k = 1;
        while (k < k_max)
            error = Y*a_temp - b_temp;
            error_save{i,j}(1,k) = sum(abs(error));
            error_plus = 0.5 * (error + abs(error));
            b_temp = b_temp + 2 * eta * error_plus;
            a_temp = inv(Y'*Y) * Y' * b_temp;
            
            k = k + 1;
        end
        
        a(i,j) = {a_temp};
        b(i,j) = {b_temp};
        
    end
    
end

end