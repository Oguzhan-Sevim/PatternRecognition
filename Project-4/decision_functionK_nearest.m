function [y] = decision_functionK_nearest(x,samples,class_length,k)

distances = sort(vecnorm(samples - x));

y = class_length^(1 / size(x,1)) / distances(k);

end