function [B_K,b_K,detB_K,B_K_invT] = affine_transformation(mesh)
%AFFINE_TRANSFORMATION
%   Calculate the affine transformation T_K*x = B_K*x + b_K and  
%   the determinant of the matrix B_K denoted as detB_K as well as the 
%   inverse of the transposed B_K_invT
%
%   INPUT:   mesh         -        mesh as a structure
%
%   OUTPUT:  B_K          -        affine transformation matrix
%            b_K          -        translation vector
%            detB_K       -        determinant of B_K
%            B_K_invT     -        the inverse of the transposed of B_K

B_K = zeros(2,2,mesh.nel);

%coordinates of the vertices A, B and C of each element
A = mesh.coordinates(:,mesh.elements(:,1));
B = mesh.coordinates(:,mesh.elements(:,2));
C = mesh.coordinates(:,mesh.elements(:,3));

%vectors spanning the triangle
a = B - A;
b = C - A;

%affine transformation transformation
B_K(:,1,:) = a;
B_K(:,2,:) = b;
b_K        = A';

% determinant
detB_K = a(1,:).*b(2,:) - a(2,:).*b(1,:);
detB_K = detB_K';    

%elements of the matrices B_K
b11 = squeeze(B_K(1,1,:)); 
b12 = squeeze(B_K(1,2,:));
b21 = squeeze(B_K(2,1,:)); 
b22 = squeeze(B_K(2,2,:));

% determinant.
det = b11.*b22 - b12.*b21;

%inverse of B_K^\top
B_K_invT = B_K;
B_K_invT(1,1,:) = b22./det;
B_K_invT(2,2,:) = b11./det;
B_K_invT(1,2,:) = - b21./det;
B_K_invT(2,1,:) = - b12./det;

end

