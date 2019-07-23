function [val,nb] = basis_dg0(p)

% Basis_DG0
%   Basis functions for the discontinuous Lagrange element on the
%   reference element
%
% Basis functions in 2D:
%       phi_1(x,y) = [1 ; 0]        phi_2(x,y) = [0 ; 1] 
%
%
% INPUT:    p         -          vector defining the points
%
% OUTPUT:   val       -          basis function values on points p
%           nb        -          the number of basis functions
%

% number of points for the basis functions to be evaluated at
dim = size(p,2);
nb = dim;

% initialize val tensor
M = size(p,1);
val  = zeros(M,dim,nb);
one = ones(M,1);          
zero = zeros(M,1);

% calculate values
val(:,:,1) = [ one  zero ];
val(:,:,2) = [ zero  one ];


end

