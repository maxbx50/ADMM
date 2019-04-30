function [out] = get_fem_matrix(mesh,type)
%GET_FEM_MATRIX
%   Calculates the FEM matrix specified by the given type on the given mesh.
%
%   INPUT:   mesh               -        mesh as a structure
%            type               -        matrix type
%
%   OUTPUT:  out                -        FEM matrix
%  
%   supported types: 
%          'mass_p1'            -        P1 mass matrix
%          'stiffness_p1'       -        P1 stiffness matrix
%          'mass_nedelec'       -        Nedelec0 mass matrix
%          'stiffness_nedelc'   -        Nedelec0 stiffness matrix
%          'B_p1dg0'            -        matrix B defined by b_ij = int_Omega inner(grad(p_i),phi_j) dx
%                                        where p_i are the P1 basis functions and psi_j the DG0 basis
%          'B_p1ned'            -        matrix B defined by b_ij = int_Omega inner(grad(p_i),phi_j) dx
%                                        where p_i are the P1 basis functions and phi_j the Nedelec0 basis functions

abs_detB_K = abs(mesh.detB_K);      
dim      = mesh.dim;               
nelems   = mesh.nel;           
B_K_invT = mesh.B_K_invT;
B_K = mesh.B_K;

switch type
    
    case('mass_p1')

        % integration points and weights
        [ip,w,nip] = intquad(2,dim);      

        %basis functions evaluated in integration points
        [val,~,nbasis] = basis_P1(ip);

        % calculate all local stiffness matrices simultaneously
        M = zeros(nbasis,nbasis,nelems);
        for i=1:nip
            for m=1:nbasis
                for k=m:nbasis   
                    M(m,k,:) = squeeze(M(m,k,:)) + ...
                                  w(i) .* abs_detB_K .* ...
                                  ( val(i,:,m) .* val(i,:,k) );
                end
            end
        end

        % copy symmetric entries 
        M = symmetrize(M);

        %gather all the local entries and build the full sparse matrix
        J = reshape(repmat(mesh.elements',nbasis,1),nbasis,nbasis,nelems);  %index sets for the global dofs
        I = permute(J,[2 1 3]);                                          
        out = sparse(I(:),J(:),M(:)); 
        
        
    case('stiffness_p1')
        % integration points and weights
        [ip,w,nip] = intquad(1,dim);       

        %basis functions evaluated in integration points
        [~,gval,nbasis] = basis_P1(ip);

        % calculate all local stiffness matrices simultaneously
        A = zeros(nbasis,nbasis,nelems);
        for i=1:nip
            for m=1:nbasis
                for k=m:nbasis 
                    A(m,k,:) = squeeze(A(m,k,:))' + ...
                                   w(i) .* abs_detB_K' .* ...
                                   sum( squeeze(tensor_vec_mult(B_K_invT, gval(i,:,m))) ...
                                        .* ...
                                        squeeze(tensor_vec_mult(B_K_invT, gval(i,:,k))) ...
                                      );
                end
            end
        end

        % copy symmetric entries 
        A = symmetrize(A);

        %gather all the local entries and build the full sparse matrix
        J = reshape(repmat(mesh.elements',nbasis,1),nbasis,nbasis,nelems);   %index sets for the global dofs
        I = permute(J,[2 1 3]);                                           
        out = sparse(I(:),J(:),A(:));                        

        
    case('mass_nedelec')
        % integration points and weights
        [ip,w,nip] = intquad(2,dim);           

        %basis functions evaluated in integration points
        [val,~,nbasis] = basis_nedelec0(ip);

        % calculate local stiffness matrices 
        M = zeros(nbasis,nbasis,nelems);
        for i=1:nip
            for m=1:nbasis
                for k=m:nbasis
                    M(m,k,:) = squeeze(M(m,k,:))' + ...
                                  w(i) .* abs_detB_K' .* ...
                                  sum( squeeze(tensor_scalar_mult(mesh.signs(:,m), tensor_vec_mult(B_K_invT, val(i,:,m))) ) ...
                                       .* ...
                                       squeeze(tensor_scalar_mult(mesh.signs(:,k), tensor_vec_mult(B_K_invT, val(i,:,k))) ) ...
                                     );
                end
            end
        end

        % copy symmetric entries
        M = symmetrize(M);

        %gather all the local entries and build the full sparse matrix
        J = reshape(repmat(mesh.edges_elements',nbasis,1),nbasis,nbasis,nelems);    %index sets for the global dofs
        I = permute(J,[2 1 3]);                                             
        out = sparse(I(:),J(:),M(:));       
        
        
    case('stiffness_nedelec')
        % integration points weights
        [ip,w,nip] = intquad(1,dim);       

        %basis functions evaluated in integration points
        [~,cval,nbasis] = basis_nedelec0(ip);

        % calculate all local stiffness matrices simultaneously
        A = zeros(nbasis,nbasis,nelems);

        %2d
        if ( dim == 2 )
            for i=1:nip
                for m=1:nbasis
                    for k=m:nbasis
                        A(m,k,:) = squeeze(A(m,k,:)) + ...
                                       w(i) .* abs_detB_K.^(-1) .* ...
                                       ( mesh.signs(:,m) .* cval(i,:,m) ) .* ...
                                       ( mesh.signs(:,k) .* cval(i,:,k) );
                    end
                end
            end

        %3d
        else
            for i=1:nip
                for m=1:nbasis
                    for k=m:nbasis
                        A(m,k,:) = squeeze(A(m,k,:))' + ...
                                       w(i) .* abs_detB_K'.^(-1) .* ...
                                       sum( squeeze(tensor_scalar_mult(mesh.signs(:,m), tensor_vec_mult(B_K, cval(i,:,m))) ) ...
                                            .* ...
                                            squeeze(tensor_scalar_mult(mesh.signs(:,k), tensor_vec_mult(B_K, cval(i,:,k))) ) ...
                                          );
                    end
                end
            end    
        end

        % copy symmetric entries 
        A = symmetrize(A);

        %gather all the local entries and build the full sparse matrix
        J = reshape(repmat(mesh.edges_elements',nbasis,1),nbasis,nbasis,nelems);    %index sets for the global dofs
        I = permute(J,[2 1 3]);                                                     
        out = sparse(I(:),J(:),A(:));     
        
    case('B_p1dg0')

        [ip,w] = intquad(1,dim);       

        %basis functions evaluated in integration points
        [val,n_dg0] = basis_dg0(ip);
        [~,gval,n_p1] = basis_P1(ip);

        % calculate all local stiffness matrices simultaneously
        B = zeros(n_p1,n_dg0,nelems);


        for m=1:n_p1
            for k=1:n_dg0
                    B(m,k,:) = squeeze(B(m,k,:))' + ...
                                  w(1) .* abs_detB_K' .* ...
                                  sum(repmat(reshape(val(1,:,k),dim,1),1,nelems)  ...
                                       .* ...
                                       squeeze(tensor_vec_mult(B_K_invT, gval(1,:,m))) ...
                                     );
            end
        end



        %gather all the local entries and build the full sparse matrix
        I = reshape(repmat(mesh.elements',n_dg0,1),n_p1,n_dg0,nelems);      %index sets for the global dofs
        J = 1:1:2*nelems;
        J = repmat(J,n_p1,1);
        out = sparse(I(:),J(:),B(:));       
        
        
    case('B_p1ned')
        
        nedges = mesh.ned;

        [ip,w] = intquad(1,dim); 

        %basis functions evaluated in integration points
        [val,~,n_ned] = basis_nedelec0(ip);
        [~,gval,n_p1] = basis_P1(ip);

        % calculate all local stiffness matrices simultaneously
        B = zeros(n_p1,n_ned,nelems);

        for m=1:n_p1
            for k=1:n_ned
                B(m,k,:) = squeeze(B(m,k,:))' + ...
                            w(1) .* abs_detB_K' .* ...
                                 sum( squeeze(tensor_scalar_mult(mesh.signs(:,k), tensor_vec_mult(B_K_invT, val(1,:,k))) ) ...
                                       .* ...
                                       squeeze(tensor_vec_mult(B_K_invT, gval(1,:,m))) ...
                                     );
            end
        end



        %gather all the local entries and build the full sparse matrix
        I = reshape(repmat(mesh.elements',n_p1,1),n_p1,n_p1,nelems);                      %index sets for the global dofs
        J = reshape(repmat(mesh.edges_elements',n_ned,1),n_ned,n_ned,nelems);  
        J = permute(J,[2,1,3]);
        out = sparse(I(:),J(:),B(:));  
        
    otherwise
        error('Unsupported matrix type.');

end
        

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