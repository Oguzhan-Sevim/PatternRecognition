function [p] = MultivariateGaussian(x,mu,sigma_det,sigma_inv,d)

p = exp( (-1/2) * (x-mu)' * sigma_inv * (x-mu) ) / ( (2*pi)^(d/2) * sigma_det^.5 ); 

end