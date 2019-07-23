function [u_vec,psnrvec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = ROF_curl(N,it,gamma,beta,imag)
%ADMM_CURL
%   ADMM method to solve the total variation problem, Y=H_curl, Y_h = N_0
%
%   INPUT:   N            -        discretization parameter for the mesh
%            it           -        maximum number of iterations
%            gamma        -        penalty parameter
%            beta         -        total variation regularization parameter
%            imag         -        image file
%
%   OUTPUT:  u_vec        -        tensor containing the iterates u_k
%            psnrvec      -        vector of PSNR values
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
tol = 1e-16;
dist = 1.0;
%it = 1;
k=0;
err = 1.0;
epsi = 1e-2;
eta = 1.0;
n_it = 1;                               % # of Newton iterations

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
r_pvec = [];
r_dvec = [];
errv = [];
gammav = gamma;
psnrvec = [];

%Read the image and generate noise
A = imread(imag);

shape = size(A);

if(length(shape)==2) %grayscale image
    
    A = double(A)/255;
    
elseif(length(shape)==3) %RGB image
    
    A = double(A);
    A = (A(:,:,1) + A(:,:,2) + A(:,:,3))/(3*255);

else
    error('Unknown image format.');
end
    
noise_level = 0.1;
noise = normrnd(0,noise_level,shape(1:2));
true_image = A;
noisy_image = A + noise;

f1 = @(x) f_imaghandle(x,true_image,shape);
f = @(x) f_imaghandle(x,noisy_image,shape);

%setup poisson problem
[~,L1] = solver_poisson(mesh, f);
[A,L2] = solver_poisson(mesh, f1);
B = get_fem_matrix(mesh,'B_p1ned');

%P1 mass matrix for error computation
M_p1 = get_fem_matrix(mesh,'mass_p1');

%Nedelec mass and stiffness matrix for d-problem
M_ned = get_fem_matrix(mesh,'mass_nedelec'); 
K_ned = get_fem_matrix(mesh,'stiffness_nedelec');

R = chol(gamma*A + M_p1);
R_T = R';

u_noisy = M_p1\L1;
u_true = M_p1\L2;

%calculate initial PSNR
MSE = norm(u_true-u_noisy)^2/512^2;
psnr = 10*log10(1/MSE);
psnrvec = [psnrvec;psnr];

    
while(dist>tol && k<it)
    k = k+1;
    fprintf('\n________________________________________________\n \n');
    fprintf('\t \t ADMM Iteration: %d \n',k);

    %-----------------------u-problem-------------------------------------
    
    L2 = B*(gamma*d_old+lamb);
    L = L1 + L2;
    u = zeros(mesh.nc,1);
    y = R_T\L;
    y1 = R\y;
    u = y1;
    
    MSE = norm(u_true-u)^2/512^2;
    psnr = 10*log10(1/MSE);
    psnrvec = [psnrvec;psnr];
                                                                                               
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
    %r_p = sqrt(2/(N^2))*sqrt((d-grad_uned)'*M_ned*(d-grad_uned));
    r_p = sqrt((d-grad_uned)'*M_ned*(d-grad_uned));
    r_d = gamma*norm(B*(d-d_old));
    fprintf('\nDistance ||u_k - u_old||_L^2 : %.7e \n', dist);
    fprintf('\nPrimal residual ||r_p^k||_L^2 : %.7e \n', r_p);
    fprintf('\nDual residual ||r_d^k||_L^2 : %.7e \n', r_d);
    
    if ((dist<1e-5) && (epsi>1e-14))
       epsi = epsi/2;       %decrease epsilon
    end
    
    %updates
    u_old = u;
    d_old = d;
    
    %store data
    u_vec(:,:,k) = u;
    d_vec(:,:,k) = d;
    lamb_vec(:,:,k) = lamb;
    distv = [distv;dist];
    r_pvec = [r_pvec;r_p];
    r_dvec = [r_dvec;r_d];
    gammav = [gammav;gamma];
end

%plot results
x = 0:1/N:1;
y = 0:1/N:1;
u_h = reshape(u_vec(:,:,end),N+1,N+1);
[xx,yy]=meshgrid(x,y);

subplot(2,2,1)
imagesc([0,0],[1,1],true_image);
title('Original image')
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca,'visible','off')
set(get(gca, 'Title'), 'Visible', 'on')

subplot(2,2,2)
imagesc([0,0],[1,1],noisy_image);
title('Noisy image')
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca,'visible','off')
set(get(gca, 'Title'), 'Visible', 'on')

subplot(2,2,3)
%s = pcolor(xx,yy,reshape(u_vec(:,:,30),N+1,N+1));
s = surf(xx,yy,reshape(u_vec(:,:,10),N+1,N+1));
set(s, 'EdgeColor', 'none');
view(0,90)
title('10 Iterations')
set(gca,'xtick',[])
set(gca,'ytick',[])

subplot(2,2,4)
s = surf(xx,yy,reshape(u_vec(:,:,end),N+1,N+1));
%imagesc(reshape(u_vec(:,:,end),N+1,N+1))
set(s, 'EdgeColor', 'none');
view(0,90)
title('50 Iterations')
set(gca,'xtick',[])
set(gca,'ytick',[])

colormap gray

end
