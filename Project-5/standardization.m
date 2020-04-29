function[x] = standardization(x)

[d,~] = size(x); % d: number of features; n: number of samples

for i = 1:d
    
    %x(i,:) = ( x(i,:) - min(x(i,:)) ) / ( max(x(i,:)) - min(x(i,:)) );
    
    x(i,:) = ( x(i,:) - mean(x(i,:)) ) / ( std(x(i,:)) );
    
end

end