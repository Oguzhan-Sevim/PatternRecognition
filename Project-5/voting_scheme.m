function[prediction_final75,prediction_final25] = voting_scheme(prediction_overall75,prediction_overall25)


[number_of_types,c] = size(prediction_overall75);

prediction_final75 = cell(1,c);

for j = 1:c
    
    for i = 1:number_of_types
        
        prediction_final75(1,j) = {[prediction_final75{1,j} ; prediction_overall75{i,j}]};
    
    end
    
    prediction_final75(1,j) = { rem(mode(prediction_final75{1,j}),c) };
    
end

prediction_final25 = cell(1,c);

for j = 1:c
    
    for i = 1:number_of_types
        
        prediction_final25(1,j) = {[prediction_final25{1,j} ; prediction_overall25{i,j}]};
    
    end
    
    prediction_final25(1,j) = { rem(mode(prediction_final25{1,j}),c) };
    
end



end