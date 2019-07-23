function [u_vec,psnrvec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = ROF_l2(N,it,gamma,beta,imag)
%ADMM_L2
%   ADMM method to solve the total variation problem, Y=L2, Y_h = DG_0
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

%ROF(512,60,0.0001,0.015/120,'Lena.png');

add_path

%Create mesh and define function space
mesh = create_unitsquaremesh(N);
nelem = mesh.nel;
npoints = mesh.nc;

%define some parameters
%beta = 0.000001;
tol = 1e-14;
dist = 1.0;
k=0;
err = 1.0;
%rho = 10;
%tau = 1.001;
%zeta = 2;

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
r_pvec = [];
r_dvec = [];
errv = [];
psnrvec = [];
gammav = gamma;

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
B = get_fem_matrix(mesh,'B_p1dg0');

%P1 mass matrix for error computation
M_p1 = get_fem_matrix(mesh,'mass_p1');
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

    %-----------------------u-problem--------------------------------------
    
    d_temp = d_old';
    d_old_vc = d_temp(:);
    lamb_temp = lamb';
    lamb_vc = lamb_temp(:);
    L2 = B*(gamma*d_old_vc+lamb_vc);  
    L = L1 + L2;
    u = zeros(mesh.nc,1);   
    y = R_T\L;
    y1 = R\y;
    u = y1;
    
    MSE = norm(u_true-u)^2/512^2;
    psnr = 10*log10(1/MSE);
    psnrvec = [psnrvec;psnr];
    
    %-----------------------d-problem--------------------------------------
     
    ip = 1/3*[1,1];         
    grad_u = eval_function(mesh, u , ip ,'P1_grad');      %evaluate grad(u) in midpoints of the triangles 
    d = shrink(grad_u-lamb/gamma,beta/gamma);
    
    %Multiplier update
    lamb = lamb - gamma * (grad_u - d);
    
    %Convergence test
    dist = sqrt((u-u_old)'*M_p1*(u-u_old));
    d_t = d';
    grad_ut = grad_u';
    r_p = sqrt(1/(2*N^2)*(d_t(:)-grad_ut(:))'*(d_t(:)-grad_ut(:)));
    r_d = gamma*norm(B*(d_t(:)-d_old_vc));
    fprintf('\nDistance ||u_k - u_old||_L^2 : %.7e \n', dist);
    fprintf('\nPrimal residual ||r_p^k||_L^2 : %.7e \n', r_p);
    fprintf('\nDual residual ||r_d^k||_L^2 : %.7e \n', r_d);

    %updating penalty parameter
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
    r_pvec = [r_pvec;r_p];
    r_dvec = [r_dvec;r_d];
    gammav = [gammav;gamma];
end

%plot results
x = 0:1/N:1;
y = 0:1/N:1;
u_h = reshape(u_vec(:,:,end),N+1,N+1);
[xx,yy]=meshgrid(x,y);
%imagesc([0,0],[1,1],u_h);

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
