function [u_vec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = admm_curl(N,f,it,gamma)
%ADMM_CURL
%   ADMM method to solve the total variation problem, Y=H_curl, Y_h = N_0
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
tol = 1e-16;
dist = 1.0;
%it = 1;
k=0;
err = 1.0;
epsi = 1e-5;
eta = 1.0;
n_it = 1;                               % # of Newton iterations
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
epsv = [];
errv = [];
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
B = get_fem_matrix(mesh,'B_p1ned');
boundary_nodes  = unique(mesh.bd_edges);
dofs            = setdiff(1:mesh.nc,boundary_nodes);
R = chol(A(dofs,dofs));
R_T = R';

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
    y = R_T\L(dofs);
    y1 = R\y;
    u(dofs) = y1/(mu+gamma);
                                                                                               
    %express grad(u) as a Nedelec function
    u_t = u(mesh.elements);
    grad_uned = zeros(nedges,1);
    grad_uned(mesh.edges_elements(:,1)) = mesh.signs(:,1).*(u_t(:,2)-u_t(:,1));
    grad_uned(mesh.edges_elements(:,3)) = mesh.signs(:,3).*(u_t(:,1)-u_t(:,3));
    grad_uned(mesh.edges_elements(:,2)) = mesh.signs(:,2).*(u_t(:,3)-u_t(:,2));
    
    
    %-----------------------d-problem-------------------------------------
    
    z = grad_uned - lamb/gamma;
    
    % prepare newton solver
    rel_error = 1;
    newt_it = 0;
    d = d_old;
    fprintf('Newton solver \n');
    
    while(rel_error > 1e-6 && newt_it < n_it)              
        newt_it = newt_it +1;
        
        %assemble the linear system jac*delta_d = -rhs
        [jac,rhs] = assemble_nl_system(mesh, d, epsi);
        jac = beta*jac + gamma*M_ned + gamma*K_ned;
        rhs = beta*rhs + gamma*M_ned*(d-z) + gamma*K_ned*(d-z);
        
        %solve the system
        delta_d = -jac\rhs;     
        rel_error = norm(delta_d);
        
        %update
        d = d + eta*delta_d;
        fnorm = norm(rhs);

        fprintf('Iteration \t absolute error \t relative error\n');
        fprintf('\t %d \t \t %.6e \t %.6e \n', newt_it , fnorm, rel_error);
          %if((newt_it==n_it) && (abs_error > 0.1*norm(d)))
           %     error('Newton method did not converge.')
          % end
    end
    
    %Multiplier update
    lamb = lamb - gamma * (grad_uned - d);
    
    
    %Convergence test
    dist = sqrt((u-u_old)'*M_p1*(u-u_old));
    %dist1 = sqrt((u-u_old)'*A*(u-u_old));
    %r_p = sqrt(2/(N^2))*sqrt((d-grad_uned)'*M_ned*(d-grad_uned));
    r_p = sqrt((d-grad_uned)'*M_ned*(d-grad_uned));
    r_d = gamma*norm(B*(d-d_old));
    fprintf('\nDistance ||u_k - u_old||_L^2 : %.7e \n', dist);
    fprintf('\nPrimal residual ||r_p^k||_L^2 : %.7e \n', r_p);
    fprintf('\nDual residual ||r_d^k||_L^2 : %.7e \n', r_d);
    
    if ((dist<1e-4) && (epsi>1e-13))
       epsi = epsi/2;       %decrease epsilon
    end
    
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
