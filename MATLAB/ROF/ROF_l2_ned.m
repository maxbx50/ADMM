function [u_vec,psnrvec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = ROF_l2_ned(N,it,gamma,beta,imag)
%ADMM_L2_NED
%   ADMM method to solve the total variation problem, Y=L2, Y_h = N_0
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
%mu = 1.0;
%gamma = 1.0;
%beta = 0.001;
tol = 1e-14;
dist = 1.0;
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
[A,L1] = solver_poisson(mesh, f);
[~,L2] = solver_poisson(mesh, f1);
%A = get_fem_matrix(mesh,'stiffness_p1');
B = get_fem_matrix(mesh,'B_p1ned');

%P1 mass matrix for error computation
M_p1 = get_fem_matrix(mesh,'mass_p1');

%Nedelec mass matrix for d-problem
M_ned = get_fem_matrix(mesh,'mass_nedelec'); 
R = chol(gamma*A + M_p1);
R_T = R';
C = chol(M_ned);
C_T = C';

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
    dist = sqrt((u-u_old)'*M_p1*(u-u_old));
    r_p = sqrt((d-grad_uned)'*M_ned*(d-grad_uned));
    r_d = gamma*norm(B*(d-d_old));
    fprintf('\nDistance ||u_k - u_old||_L^2 : %.7e \n', dist);
    fprintf('\nPrimal residual ||r_p^k||_L^2 : %.7e \n', r_p);
    fprintf('\nDual residual ||r_d^k||_L^2 : %.7e \n', r_d);
    
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

