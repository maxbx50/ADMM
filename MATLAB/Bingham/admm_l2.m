function [u_vec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = admm_l2(N,f,it,gamma)
%ADMM_L2
%   ADMM method to solve the total variation problem, Y=L2, Y_h = DG0
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

%define some parameters
%mu = 10.0;
mu = 1.0;
%gamma = 1.0;
beta = 0.8;
%beta = 0.1;
%beta = 0.001;
tol = 1e-14;
dist = 1.0;
k=0;
err = 1.0;
rho = 5;
tau = 1.1;
zeta = 1/3;

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
distv1 = [];
r_pvec = [];
r_dvec = [];
errv = [];
gammav = gamma;

%Define linear variational problem
%a = (mu+gamma)*inner(grad(u), grad(v))*dx
%L = f*v*dx+inner(gamma*d+lamb, grad(v))*dx

%setup poisson problem
[A,L1] = solver_poisson(mesh, f);
%A = get_fem_matrix(mesh,'stiffness_p1');
B = get_fem_matrix(mesh,'B_p1dg0');
boundary_nodes    = unique(mesh.bd_edges);
dofs            = setdiff(1:mesh.nc,boundary_nodes);
%L_inc = ichol(A(dofs,dofs));
R = chol(A(dofs,dofs));
R_T = R';
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
    y = R_T\L(dofs);
    y1 = R\y;
    %y1 = pcg(A(dofs,dofs),L(dofs),1e-6,150,L_inc,L_inc');
    u(dofs) = y1/(mu+gamma);
    
    %-----------------------d-problem--------------------------------------
     
    ip = 1/3*[1,1];         
    grad_u = eval_function(mesh, u , ip ,'P1_grad');      %evaluate grad(u) in midpoints of the triangles 
    d = shrink(grad_u-lamb/gamma,beta/gamma);
    
    %Multiplier update
    lamb = lamb - gamma * (grad_u - d);
    
    %Convergence test
    dist = sqrt((u-u_old)'*M_p1*(u-u_old));       %distance in L_2 norm
    %dist = sqrt((u-u_old)'*A*(u-u_old));         %distance in H_1 norm
    d_t = d';
    grad_ut = grad_u';
    r_p = sqrt(1/(2*N^2)*(d_t(:)-grad_ut(:))'*(d_t(:)-grad_ut(:)));
    r_d = gamma*norm(B*(d_t(:)-d_old_vc));
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

