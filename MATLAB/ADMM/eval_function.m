function [out] = eval_function(mesh,coeff_vector,P,function_type)
%EVAL_FUNCTION
%   Function to evaluate a mesh function at given points
%
% INPUT:    mesh             -         mesh as a structure
%           coeff_vector     -         coefficient vector of the mesh function
%           P                -         evaluation points
%           function_type    -         type of the mesh function
%
% OUTPUT:   out              -         function values in the given points
%

nelem = mesh.nel;
B_K_invT = mesh.B_K_invT;

switch function_type 
    
    case('P1')
            u_coeff_local = coeff_vector(mesh.elements);
            val = basis_P1(P);
    
            for i = 1:size(P,1)
             out(:,:,i) = u_t(:,1).*val(i,:,1) + u_t(:,2).*val(i,:,2) + u_t(:,3).*val(i,:,3);
            end
            
        
    case('P1_grad')
            u_coeff_local = coeff_vector(mesh.elements);
            u_coeff_local = reshape(u_coeff_local',1,size(mesh.elements,2),nelem);
            [~,gval] = basis_P1(P);
    
            for i = 1:size(P,1)
             out(:,:,i) = u_coeff_local(1,1,:).*tensor_vec_mult(B_K_invT, gval(i,:,1)) + u_coeff_local(1,2,:).*tensor_vec_mult(B_K_invT, gval(i,:,2)) ...
                        + u_coeff_local(1,3,:).*tensor_vec_mult(B_K_invT, gval(i,:,3));
            end
            
            out = reshape(permute(out,[3,2,1]),nelem,2);
        
    case('Nedelec')
            val = basis_nedelec0(P);
            
            for i = 1:size(P,1)
                out =  coeff_vector(mesh.edges_elements(:,1))'.*squeeze(tensor_scalar_mult(mesh.signs(:,1), tensor_vec_mult(B_K_invT, val(i,:,1)))) + ...
                    coeff_vector(mesh.edges_elements(:,2))'.*squeeze(tensor_scalar_mult(mesh.signs(:,2), tensor_vec_mult(B_K_invT, val(i,:,2)))) + ... 
                    coeff_vector(mesh.edges_elements(:,3))'.*squeeze(tensor_scalar_mult(mesh.signs(:,3), tensor_vec_mult(B_K_invT, val(i,:,3))));
            end    
        
            out = out';
           
    %case('Nedelec_curl')
    otherwise
        error('Unsupported function type.');
        
end


end

