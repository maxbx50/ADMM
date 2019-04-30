function [u_vec,d_vec,lamb_vec,distv] = test(N,f,it,gamma)
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

%initialize vectors
u_vec = zeros(npoints,1,it);
d_vec = zeros(nedges,1,it);
lamb_vec = zeros(nedges,1,it);
u_old = zeros(npoints,1);
lamb = zeros(nedges,1);
d_old = zeros(nedges,1);
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
B = matrix_dp1_ned(mesh);
boundary_nodes    = unique(mesh.bd_edges);
dofs            = setdiff(1:mesh.nc,boundary_nodes);

%P1 mass matrix for error computation
M_p1 = mass_matrix_P1(mesh);

%Nedelec mass matrix for d-problem
M_ned = mass_matrix_Nedelec0(mesh);

while(dist>tol && k<it)
    k = k+1;
    fprintf('\n________________________________________________\n \n');
    fprintf('\t \t ADMM Iteration: %d \n',k);

     %-----------------------u-problem-------------------------------------
     
    L2 = B*(gamma*d_old+lamb);
    L = L1 + L2;
    u = zeros(mesh.nc,1);
    u(dofs) = K_p1(dofs,dofs)\L(dofs);
    
    %Convergence test
    %print('Norm of d:', sqrt(assemble(inner(d, d) * dx)))
    dist = sqrt((u-u_old)'*M_p1*(u-u_old));
    fprintf('\nDistance ||u_k - u_old||_L^2 : %.7e \n', dist);
    distv = [distv;dist];

    %express grad(u) as a Nedelec function
    u_t = u(mesh.elements);
    grad_uned = zeros(nedges,1);
    grad_uned(mesh.edges_elements(:,1))=mesh.signs(:,1).*(u_t(:,2)-u_t(:,1));
    grad_uned(mesh.edges_elements(:,3))=mesh.signs(:,3).*(u_t(:,1)-u_t(:,3));
    grad_uned(mesh.edges_elements(:,2))=mesh.signs(:,2).*(u_t(:,3)-u_t(:,2));
    
    z = grad_uned - lamb/gamma;
    
    d = zeros(nedges,1);
    d(mesh.edges_elements(:,1)) = z(mesh.edges_elements(:,1));
    d(mesh.edges_elements(:,3)) = z(mesh.edges_elements(:,3));
    d(mesh.edges_elements(:,2)) = -z(mesh.edges_elements(:,1))-z(mesh.edges_elements(:,3));
    
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