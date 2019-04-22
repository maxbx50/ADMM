function [u_vec,d_vec,lamb_vec,distv] = admm_curl(N,f,it,gamma)
%ADMM_CURL
%   ADMM method to solve the total variation denoising problem
%
%   INPUT:   N            -        discretization parameter for the mesh
%            f            -        function handle for the source term
%            it           -        maximum number of iterations
%
%   OUTPUT:  u_vec        -        tensor containing the iterates u_k
%            d_vec        -        tensor containing the iterates d_k
%            lamb_vec     -        tensor containing the iterates lamb_k
%            distv        -        vector containing the distances ||u_k+1 - u_k||_L^2

%add_path

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
%it = 1;
k=0;
err = 1.0;
epsi = 1e-5;
eta = 1.0;
n_it = 20;                               % # of Newton iterations

%initialize vectors
u_vec = zeros(npoints,1,it);
d_vec = zeros(nedges,1,it);
lamb_vec = zeros(nedges,1,it);
u_old = zeros(npoints,1);
lamb = zeros(nedges,1);
d_old = zeros(nedges,1);
epsv = [];
distv = [];
errv = [];


%Define linear variational problem
%a = (mu+gamma)*inner(grad(u), grad(v))*dx
%L = f*v*dx+inner(gamma*d+lamb, grad(v))*dx

%setup poisson problem
[K_p1,L1] = solver_poisson(mesh, f);
K_p1 = (mu+gamma)*K_p1;
B = get_fem_matrix(mesh,'B_p1ned');
boundary_nodes  = unique(mesh.bd_edges);
dofs            = setdiff(1:mesh.nc,boundary_nodes);

%P1 mass matrix for error computation
M_p1 = get_fem_matrix(mesh,'mass_p1');

%Nedelec mass and stiffness matrix for d-problem
M_ned = get_fem_matrix(mesh,'mass_nedelec'); 
K_ned = get_fem_matrix(mesh,'stiffness_nedelec');

    
while(dist>tol && k<it)
    k = k+1;
    fprintf('\n________________________________________________\n \n');
    fprintf('\t \t ADMM Iteration: %d \n',k);

    %-----------------------u-problem-------------------------------------
    
    L2 = B*(gamma*d_old+lamb);
    L = L1 + L2;
    u = zeros(mesh.nc,1);
    u(dofs) = K_p1(dofs,dofs)\L(dofs);
                                                                                               
    %express grad(u) as a Nedelec function
    u_t = u(mesh.elements);
    grad_uned = zeros(nedges,1);
    grad_uned(mesh.edges_elements(:,1)) = mesh.signs(:,1).*(u_t(:,2)-u_t(:,1));
    grad_uned(mesh.edges_elements(:,3)) = mesh.signs(:,3).*(u_t(:,1)-u_t(:,3));
    grad_uned(mesh.edges_elements(:,2)) = mesh.signs(:,2).*(u_t(:,3)-u_t(:,2));
    
    
    %-----------------------d-problem-------------------------------------
    
    z = grad_uned - lamb/gamma;
    
    % prepare newton solver
    abs_error = 1;
    newt_it = 0;
    d = d_old;
    fprintf('Newton solver \n');
    
    while(abs_error > 1e-6 && newt_it < n_it)              
        newt_it = newt_it +1;
        
        %assemble the linear system jac*delta_d = -rhs
        [jac,rhs] = assemble_nl_system(mesh, d, epsi);
        jac = beta*jac + gamma*M_ned + gamma*K_ned;
        rhs = beta*rhs + gamma*M_ned*(d-z) + gamma*K_ned*(d-z);
        
        %solve the system
        delta_d = -jac\rhs;     
        abs_error = norm(delta_d);
        
        %update
        d = d + eta*delta_d;
        fnorm = norm(rhs);

        fprintf('Iteration \t absolute error \t relative error\n');
        fprintf('\t %d \t \t %.6e \t %.6e \n', newt_it , abs_error, fnorm);
           if((newt_it==n_it) && (abs_error > 0.1*norm(d)))
                error('Newton method did not converge.')
           end
    end
    
    %Convergence test
    %print('Norm of d:', sqrt(assemble(inner(d, d) * dx)))
    dist = sqrt((u-u_old)'*M_p1*(u-u_old));
    fprintf('\nDistance ||u_k - u_old||_L^2 : %.7e \n', dist);
    distv = [distv;dist];
    if ((dist<1e-4) && (epsi>1e-13))
       epsi = epsi/2;       %decrease epsilon
    end
    
    %updates
    u_old = u;
    d_old = d;
    lamb = lamb - gamma * (grad_uned - d);
    
    %store data
    u_vec(:,:,k) = u;
    d_vec(:,:,k) = d;
    lamb_vec(:,:,k) = lamb;
end

end
