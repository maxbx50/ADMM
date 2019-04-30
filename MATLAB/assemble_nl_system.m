function [G,rhs] = assemble_nl_system(mesh, d,eps1)
%ASSEMBLE_NL_SYSTEM
%   Assembles the jacobi matrix for the total variation term of the regularized
%   total variation problem
%   
%
%   INPUT:   mesh      -        mesh as a structure
%            d         -        coefficient vector of d_h
%            eps1      -        regularization parameter
%
%   OUTPUT:  G         -        jacobi matrix of the total variation term
%            rhs       -        total variation evaluated at previous iterate


dim      = mesh.dim;           % the dimension of the problem
nelems   = mesh.nel;      % number of elements
abs_detB_K = abs(mesh.detB_K);
B_K_invT = mesh.B_K_invT;

[ip,w,nip] = intquad(2,dim);         % integration points and weights

%basis functions evaluated in integration points
[val,~,nbasis] = basis_nedelec0(ip);

% calculate all local stiffness matrices simultaneously
G = zeros(nbasis,nbasis,nelems);
rhs = zeros(nelems,nbasis);

for i=1:nip
    d_eval = eval_function(mesh,d,ip(i,:),'Nedelec');
    [f,J_eval] = fun_jac(d_eval,nelems,eps1);
    
    for m=1:nbasis
        for k=m:nbasis
            temp1 = squeeze(tensor_scalar_mult(mesh.signs(:,m), tensor_vec_mult(B_K_invT, val(i,:,m))));
            temp1 = sum(J_eval.*reshape(repmat(temp1(:),1,dim)',dim,dim,nelems),2);
            temp2 = tensor_scalar_mult(mesh.signs(:,k), tensor_vec_mult(B_K_invT, val(i,:,k)));
            
            G(m,k,:) = squeeze(G(m,k,:))' + w(i) .* abs_detB_K' .* squeeze(sum( temp1.*temp2 ))';
        end
    end
    
    for j=1:nbasis
        rhs(:,j) = rhs(:,j) + ...
            	    w(i) .* abs_detB_K .* ...
                    sum( squeeze(tensor_scalar_mult(mesh.signs(:,j), tensor_vec_mult(B_K_invT, val(i,:,j)))).* f')';
    end
    
end

% copy symmetric entries of the local matrices
G = symmetrize(G);

%gather all the local entries and build the full sparse matrix
J = reshape(repmat(mesh.edges_elements',nbasis,1),nbasis,nbasis,nelems);    %index sets for the global dofs
I = permute(J,[2 1 3]);                                          
G = sparse(I(:),J(:),G(:));   
rhs = sparse(mesh.edges_elements,1,rhs);  

end

function [rhs,jac] = fun_jac(d,nelem,eps1)

rhs = d./ sqrt(d(:,1).*d(:,1)+d(:,2).*d(:,2) + eps1);

A = zeros(size(d,2),size(d,2),nelem);

A(1,1,:) = d(:,2).^2 + eps1;
A(2,1,:) = -d(:,2).*d(:,1);
A(1,2,:) = -d(:,2).*d(:,1);
A(2,2,:) = d(:,1).^2 + eps1;

jac = A./reshape(sqrt(d(:,1).*d(:,1)+d(:,2).*d(:,2) + eps1).^3,1,1,nelem);

end

function A = symmetrize(A)
%   Copy the upper triangular part of square matrices in the tensor 
%   A(:,:,i) such that the result are symmetric matrices.

m = size(A,1);
n = size(A,2);

for i=1:m
    for j=i+1:n
        A(j,i,:) = A(i,j,:);
    end
end

end
