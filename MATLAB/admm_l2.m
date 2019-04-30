function [u_vec,d_vec,lamb_vec,distv,mesh] = admm_l2(N,f,it,gamma)
%ADMM_L2
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

%define some parameters
mu = 1.0;
%gamma = 1.0;
beta = 0.8;
tol = 1e-14;
dist = 1.0;
%it = 1;
k=0;
err = 1.0;

%initialize vectors
u_vec = zeros(npoints,1,it);
d_vec = zeros(nelem,2,it);
lamb_vec = zeros(nelem,2,it);
u_old = zeros(npoints,1);
lamb = zeros(nelem,2);
d_old = zeros(nelem,2);
b = lamb/gamma;
epsv = [];
distv = [];
errv = [];

%Define linear variational problem
%a = (mu+gamma)*inner(grad(u), grad(v))*dx
%L = f*v*dx+inner(gamma*d+lamb, grad(v))*dx

%setup poisson problem
[K_p1,L1] = solver_poisson(mesh, f);
K_p1 = (mu+gamma)*K_p1;
B = get_fem_matrix(mesh,'B_p1dg0');
boundary_nodes    = unique(mesh.bd_edges);
dofs            = setdiff(1:mesh.nc,boundary_nodes);

%P1 mass matrix for error computation
M_p1 = get_fem_matrix(mesh,'mass_p1');

while(dist>tol && k<it)
    k = k+1;
    fprintf('\n________________________________________________\n \n');
    fprintf('\t \t ADMM Iteration: %d \n',k);

    %-----------------------u-problem--------------------------------------
    
    d_temp = d_old';
    d_old_vc = d_temp(:);
    lamb_temp = lamb';
    lamb_vc = lamb_temp(:);
    L2 = B*(gamma*d_old_vc+lamb_vc);
    L = L1 + L2;
    u = zeros(mesh.nc,1);
    u(dofs) = K_p1(dofs,dofs)\L(dofs);

    %-----------------------d-problem--------------------------------------
     
    ip = 1/3*[1,1];         
    grad_u = eval_function(mesh, u , ip ,'P1_grad');      %evaluate grad(u) in midpoints of the triangles 
    d = shrink(grad_u-lamb/gamma,beta/gamma);
    
    %Convergence test
    %print('Norm of d:', sqrt(assemble(inner(d, d) * dx)))
    dist = sqrt((u-u_old)'*M_p1*(u-u_old));
    fprintf('\nDistance ||u_k - u_old||_L^2 : %.7e \n', dist);

    %updates
    u_old = u;
    d_old = d;
    lamb = lamb - gamma * (grad_u - d);
    
    %store data
    u_vec(:,:,k) = u;
    d_vec(:,:,k) = d;
    lamb_vec(:,:,k) = lamb;
    distv = [distv;dist];
end

end

