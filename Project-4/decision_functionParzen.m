function [y] = decision_functionParzen(x,samples,class_length)

y = 0;

for i = 1: size(samples,2)
    
   y = y + class_length * exp( -(x-samples(:,i))'*(x-samples(:,i)) / 2 );
    
end

end