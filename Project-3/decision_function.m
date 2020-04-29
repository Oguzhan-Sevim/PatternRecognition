function [y] = decision_function(x,mu,sigma_det,sigma_inv,class_length)

y = ( (x-mu)' * sigma_inv * (x-mu) ) + log(sigma_det) - 2*log(class_length);


end