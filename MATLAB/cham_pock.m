function [u_vec,d_vec,u_bar_vec,distv,mesh] = cham_pock(N,f,it)
%CHAM_POCK
%   Chambolle's and Pock's method to solve the total variation denoising problem
%
%   INPUT:   N            -        discretization parameter for the mesh
%            f            -        function handle for the source term
%            it           -        maximum number of iterations
%
%   OUTPUT:  u_vec        -        tensor containing the iterates u_k
%            d_vec        -        tensor containing the iterates d_k
%            distv        -        vector containing the distances ||u_k+1 - u_k||_L^2

add_path
%Create mesh and define function space
mesh = create_unitsquaremesh(N);
nelem = mesh.nel;
npoints = mesh.nc;

%define some parameters
mu = 1.0;
%tau = 10^25;
%sigma = 1.0/(sqrt(7.8)*tau);
tau = 1.0;
sigma= 1.0;
theta = 1.0;
beta = 0.8;
tol = 1e-14;
dist = 1.0;
k=0;
err = 1.0;

%initialize vectors
u_vec = zeros(npoints,1,it);
d_vec = zeros(nelem,2,it);
u_bar_vec = zeros(npoints,1,it);
u_old = zeros(npoints,1);
d_old = zeros(nelem,2);
u_bar = zeros(nelem,2);
epsv = [];
distv = [];
errv = [];


%setup poisson problem
[K_p1,L1] = solver_poisson(mesh, f);
B = get_fem_matrix(mesh,'B_p1dg0');
boundary_nodes    = unique(mesh.bd_edges);
dofs            = setdiff(1:mesh.nc,boundary_nodes);

%P1 mass matrix for error computation
M_p1 = get_fem_matrix(mesh,'mass_p1');

while(dist>tol && k<it)
    k = k+1;
    fprintf('\n________________________________________________\n \n');
    fprintf('\t \t Iteration: %d \n',k);
    
    %-----------------------d-problem--------------------------------------
    
    ip = 1/3*[1,1];         
    grad_u_bar = eval_function(mesh, u_bar , ip ,'P1_grad'); 
    d_new = sigma*grad_u_bar + d_old;
    norm_d = sqrt(sum(d_new.*d_new,2));
    idd = norm_d > beta;
    d_new(idd,:) = beta*d_new(idd,:)./norm_d(idd);
    

    %-----------------------u-problem--------------------------------------
    
    d_temp = d_new';
    d_new_vc = d_temp(:);
    L2 = B*d_new_vc;
    L = tau*L1 + K_p1*u_old - tau*L2;
    u = zeros(mesh.nc,1);
    u(dofs) = ((1+tau*mu)*K_p1(dofs,dofs))\L(dofs);
    
    %theta = 1 / sqrt(1 + 2 * mu * tau);
    %tau = theta * tau;
    %sigma = sigma/theta;
     
    %Convergence test
    %print('Norm of d:', sqrt(assemble(inner(d, d) * dx)))
    u_dist = u-u_old;
    dist = sqrt(u_dist'*M_p1*u_dist);
    fprintf('\nDistance ||u_k - u_old||_L^2 : %.7e \n', dist);

    %updates
    u_old = u;
    d_old = d_new;
    u_bar = u + theta*u_dist;
    
    %store data
    u_vec(:,:,k) = u;
    d_vec(:,:,k) = d_new;
    u_bar_vec(:,:,k) = u_bar;
    distv = [distv;dist];
    
end

end

