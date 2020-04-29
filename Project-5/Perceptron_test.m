function[prediction_overall75,prediction_overall25] = Perceptron_test(x_75_by_type_and_classes,x_25_by_type_and_classes,a)

prediction_overall75 = cell(size(x_75_by_type_and_classes));
prediction_overall25 = cell(size(x_25_by_type_and_classes));

for i = 1:size(x_75_by_type_and_classes,1)
    for j = 1:size(x_75_by_type_and_classes,2)
        data = [ x_75_by_type_and_classes{i,j} ; ones(1, size(x_75_by_type_and_classes{i,j},2)) ]';     
        prediction_overall75(i,j) = {zeros(size(x_75_by_type_and_classes,2),size(data,1))};
        for k = 1:size(data,1)
            for l = 1:size(x_75_by_type_and_classes,2)
            prediction_overall75{i,j}(l,k) = data(k,:)*a{i,l};
            end
        end
        prediction_overall75{i,j} = rem(find(prediction_overall75{i,j}==max(prediction_overall75{i,j})),size(x_75_by_type_and_classes,2))';
    end
end

for i = 1:size(x_25_by_type_and_classes,1)
    for j = 1:size(x_25_by_type_and_classes,2)
        data = [ x_25_by_type_and_classes{i,j} ; ones(1, size(x_25_by_type_and_classes{i,j},2)) ]';     
        prediction_overall25(i,j) = {zeros(size(x_25_by_type_and_classes,2),size(data,1))};
        for k = 1:size(data,1)
            for l = 1:size(x_75_by_type_and_classes,2)
            prediction_overall25{i,j}(l,k) = data(k,:)*a{i,l};
            end
        end
        prediction_overall25{i,j} = rem(find(prediction_overall25{i,j}==max(prediction_overall25{i,j})),size(x_25_by_type_and_classes,2))';
    end
end

end