function [val,cval,nb] = basis_nedelec0(p)
% BASIS_NEDELEC0
%   Basis functions for Nedelec's edge element of lowest order on the
%   reference element
%
% INPUT:    p           -         evaluation points
%
% OUTPUT:   val         -         basis function values
%           nb          -         number of basis functions
%
% Basis functions in 2D
%
%   phi_1(x,y) = [ 1 - y ; x ]     phi_2(x,y) = [ -y ; x] 
%   phi_3(x,y) = [ -y ; x - 1]
    
nb = 3;

M    = size(p,1);
val  = zeros( M , 2 , 3 );
cval = zeros( M , 1 , 3 );

% basis function values
val(:,:,1) = [ 1-p(:,2)   p(:,1)   ];
val(:,:,2) = [ -p(:,2)    p(:,1)   ];
val(:,:,3) = [ -p(:,2)    p(:,1)-1 ];

% curl values
cval = cval + 2;
    
end