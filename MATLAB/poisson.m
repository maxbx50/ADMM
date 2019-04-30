function [u_h] = poisson(mesh,f)
%POISSON
%   solves the Poisson problem with zero Dirichlet BC
%
%   INPUT:   mesh      -        mesh as a structure
%            f         -        function handle for source term
%
%   OUTPUT:  u_h       -        solution as mesh function 
%

[A,b] = solver_poisson(mesh,f);

dofs            = setdiff(1:mesh.nc,unique(mesh.bd_edges));
u_h             = zeros(mesh.nc,1);
u_h(dofs)       = A(dofs,dofs) \ b(dofs);


end

