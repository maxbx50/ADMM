function [A,L] = solver_poisson(mesh, f)

%   Poisson solver i.e. solves the problem 
%
%            int inner(grad(u),grad(v))*dx  =  int f*v d*x 
%
%   numerically on a given mesh.
%
% 	INPUT:   mesh      -        mesh as a structure
%            f         -        source term
%
%   OUTPUT:  A         -        stiffness matrix
%            L         -        load vector
%


%affine transformations
B_K = mesh.B_K;
b_K = mesh.b_K;
abs_detB_K = abs(mesh.detB_K);
dim      = mesh.dim;
nelems = mesh.nel;

%stiffness matrix
A = get_fem_matrix(mesh,'stiffness_p1');

%integration points (order 5)
[ip,w,nip] = intquad(5,dim);       

% basis functions evaluated in integration points
[val,~,nb] = basis_P1(ip);

%load vector
L = zeros(nelems,nb);

for i=1:nip
    %affine transformation of the integration points
    T_ip = squeeze(tensor_vec_mult(B_K, ip(i,:)))' + b_K;
    
    %source term evaluated in int points
    fval = f(T_ip);
    
    for k=1:nb
        L(:,k) = L(:,k) + w(i) .* abs_detB_K .* (fval .* val(i,:,k));
    end
end

%collect the local contributions for the full vector
L = sparse(mesh.elements,1,L); 


end