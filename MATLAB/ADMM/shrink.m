function d = shrink(z,b)
%SHRINK
%Shrinkage function
%   
%   INPUT:   z         -        vector to shrink
%            b         -        amount to shrink
%
%   OUTPUT:  d         -        shrinked vector
%

d = zeros(size(z,1),size(z,2));

norm_z = sqrt(sum(z.*z,2));
red_norm = norm_z-b;
idx = red_norm>0;
d(idx,:) = z(idx,:)./norm_z(idx).*red_norm(idx);
    
end