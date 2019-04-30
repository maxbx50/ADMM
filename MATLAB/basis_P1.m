function [val,gval,nb] = basis_P1(p)
% BASIS_P1
%   Basis functions for Nedelec's edge element of lowest order on the
%   reference element
%
%   INPUT:    p           -         evaluation points
%
%   OUTPUT:   val         -         basis function values
%             gval        -         gradient values
%             nb          -         number of basis functions
%
%   Basis functions in 2D
%
%   phi_1(x,y) = 1 - x - y      phi_2(x,y) = x 
%   phi_3(x,y) = y

dim = size(p,2);


nb = 3;

% initialize val and gval tensor
M = size(p,1);
val  = zeros(M,1,nb);
gval = zeros(M,2,nb);
one = ones(M,1);        
zero = zeros(M,1);

% calculate basis function values
val(:,:,1) = -p(:,1)-p(:,2)+1;
val(:,:,2) = p(:,1);
val(:,:,3) = p(:,2);

% calculate gradient(basis function) values
gval(:,:,1) = [ -one  -one ];
gval(:,:,2) = [  one   zero ];
gval(:,:,3) = [  zero   one ];
    

    
end

