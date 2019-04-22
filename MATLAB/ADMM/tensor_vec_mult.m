function out = tensor_vec_mult(T,rhs)
% Computes the matrix vector product of the nz frontal slices 
% of the tensor T with the vector rhs 

% INPUT:   T         -         Tensor of dimension [nx,ny,nz]        
%          rhs       -         right hand side of dimension [ny,1]
%
% OUTPUT   out       -         matrix vector product

[nx,ny,nz] = size(T);

%Create a tensor of the same shape as T
rhs = reshape(rhs,1,ny);
rhs = rhs(ones(nx,1),1:ny,ones(nz,1));

%Hadamard product
rhs = T .* rhs;

%sum entries rowwise
out = sum(rhs,2);

end