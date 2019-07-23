function out = tensor_scalar_mult(scalar_vec,T)
% Multiplies each frontal slice T_j, j=1:nz of the tensor T with the corresponding 
% scalar in scalar_Vec
%   
%   INPUT:   T              -        Tensor
%            scalar_vec     -        vector containing the scalars
%
%   OUTPUT:  out            -        result of the operation
%

[nx,ny,nz] = size(T);

%build a tensor of scalar_vec of the same structure as T
scalar_vec = repmat(scalar_vec',nx,1);
scalar_vec = reshape(scalar_vec,nx,ny,nz);

%Hadamard product
out = scalar_vec .* T;

end