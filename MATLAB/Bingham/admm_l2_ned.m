function [u_vec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = admm_l2_ned(N,f,it,gamma)
%ADMM_L2_NED
%   ADMM method to solve the total variation problem, Y=L2, Y_h = N_0
%
%   INPUT:   N            -        discretization parameter for the mesh
%            f            -        function handle for the source term
%            it           -        maximum number of iterations
%            gamma        -        penalty parameter
%
%   OUTPUT:  u_vec        -        tensor containing the iterates u_k
%            d_vec        -        tensor containing the iterates d_k
%            lamb_vec     -        tensor containing the iterates lamb_k
%            distv        -        vector containing the distances ||u_k+1 - u_k||_L^2
%            r_pvec       -        vector containing the primal residuals
%            r_dvec       -        vector containing the dual residuals
%            mesh         -        mesh as a structure
%            gammav       -        vector containing the penalty parameters (for an adapative choice)

add_path

%Create mesh and define function space
mesh = create_unitsquaremesh(N);
nelem = mesh.nel;
npoints = mesh.nc;
nedges = mesh.ned;

%define some parameters
mu = 1.0;
%gamma = 1.0;
beta = 0.8;
tol = 1e-14;
dist = 1.0;
k=0;
err = 1.0;
rho = 5;
tau = 1.1;
zeta = 1/3;

%initialize vectors
u_vec = zeros(npoints,1,it);
d_vec = zeros(nedges,1,it);
lamb_vec = zeros(nedges,1,it);
u_old = zeros(npoints,1);
lamb = zeros(nedges,1);
d_old = zeros(nedges,1);
b = lamb/gamma;
distv = [];
distv1 = [];
r_pvec = [];
r_dvec = [];
errv = [];
gammav = gamma;

%setup poisson problem
[A,L1] = solver_poisson(mesh, f);
B = get_fem_matrix(mesh,'B_p1ned');
boundary_nodes    = unique(mesh.bd_edges);
dofs            = setdiff(1:mesh.nc,boundary_nodes);
R = chol(A(dofs,dofs));
R_T = R';

%P1 mass matrix for error computation
M_p1 = get_fem_matrix(mesh,'mass_p1');

%Nedelec mass matrix for d-problem
M_ned = get_fem_matrix(mesh,'mass_nedelec'); 
C = chol(M_ned);
C_T = C';

while(dist>tol && k<it)
    k = k+1;
    fprintf('\n________________________________________________\n \n');
    fprintf('\t \t ADMM Iteration: %d \n',k);

     %-----------------------u-problem-------------------------------------
     
    L2 = B*(gamma*d_old+lamb);
    L = L1 + L2;
    u = zeros(mesh.nc,1);
    y = R_T\L(dofs);
    y1 = R\y;
    u(dofs) = y1/(mu+gamma);
    

    %-----------------------d-problem--------------------------------------
    [ip,w,nip] = intquad(2,mesh.dim);     
    
    
    %get the load vector
    [val,~,nbasis] = basis_nedelec0(ip);

    L_ned = zeros(nelem,nbasis);
    
    for j=1:nip
      grad_u = eval_function(mesh, u , ip(j,:) ,'P1_grad');                  %evaluate grad(u) and lambda in midpoints 
      lamb_eval = eval_function(mesh, lamb, ip(j,:) ,'Nedelec');
      shrinked_gradient = shrink(grad_u-lamb_eval/gamma,beta/gamma);
    
      for i=1:nbasis
        L_ned(:,i) = L_ned(:,i) + w(j) .* abs(mesh.detB_K) .* ...
                    sum( squeeze(tensor_scalar_mult(mesh.signs(:,i), tensor_vec_mult(mesh.B_K_invT, val(j,:,i)))) ... 
                    .* shrinked_gradient')';
      end
    
    end
      
    %Solve the best-approximation problem min_d 1/2||d-shrink||_L2^2
    d_help = C_T \ sparse(mesh.edges_elements,1,L_ned);
    d = C \ d_help;    

    %express grad(u) as a Nedelec function
    u_t = u(mesh.elements);
    grad_uned = zeros(nedges,1);
    grad_uned(mesh.edges_elements(:,1)) = mesh.signs(:,1).*(u_t(:,2)-u_t(:,1));
    grad_uned(mesh.edges_elements(:,3)) = mesh.signs(:,3).*(u_t(:,1)-u_t(:,3));
    grad_uned(mesh.edges_elements(:,2)) = mesh.signs(:,2).*(u_t(:,3)-u_t(:,2));
    
    %Multiplier update
    lamb = lamb - gamma * (grad_uned - d);
    
    %Convergence test
    dist = sqrt((u-u_old)'*M_p1*(u-u_old));          %distance in L_2 norm
    %dist = sqrt((u-u_old)'*A*(u-u_old));            %distance in H_1 norm
    r_p = sqrt((d-grad_uned)'*M_ned*(d-grad_uned));
    r_d = gamma*norm(B*(d-d_old));
    fprintf('\nDistance ||u_k - u_old||_L^2 : %.7e \n', dist);
    fprintf('\nPrimal residual ||r_p^k||_L^2 : %.7e \n', r_p);
    fprintf('\nDual residual ||r_d^k||_L^2 : %.7e \n', r_d);
    
    %Adaptive penalty parameter
    %if(r_p>rho*zeta*r_d)
    %   gamma = gamma*tau;
    %elseif(r_d>rho/zeta*r_p)
    %    gamma = gamma/tau;    
    %end
    
    %updates
    u_old = u;
    d_old = d;
    
    %store data
    u_vec(:,:,k) = u;
    d_vec(:,:,k) = d;
    lamb_vec(:,:,k) = lamb;
    distv = [distv;dist];
    %distv1 = [distv1;dist1];
    r_pvec = [r_pvec;r_p];
    r_dvec = [r_dvec;r_d];
    gammav = [gammav;gamma];
end

end
