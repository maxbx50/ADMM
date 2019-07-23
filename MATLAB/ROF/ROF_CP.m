function [u_vec,psnrvec,d_vec,u_bar_vec,distv,mesh] = ROF_CP(N,it,beta,imag)
%CHAM_POCK
%   Chambolle's and Pock's method to solve the total variation problem
%
%   INPUT:   N            -        discretization parameter for the mesh
%            it           -        maximum number of iterations
%            beta         -        total variation regularization parameter
%            imag         -        image file         
%
%   OUTPUT:  u_vec        -        tensor containing the iterates u_k
%            psnrvec      -        vector of PSNR values
%            d_vec        -        tensor containing the iterates d_k
%            u_bar_vec    -        tensor containing the iterates u_bar
%            distv        -        vector containing the distances ||u_k+1 - u_k||_L^2
%            mesh         -        mesh as a structure

add_path
%Create mesh and define function space
mesh = create_unitsquaremesh(N);
nelem = mesh.nel;
npoints = mesh.nc;

%define some parameters
%mu = 1.0;
sigma = 0.001;
tau = 0.0009999/sigma;
%tau = 0.5;
%sigma= 2;
theta = 10.0;
%beta = 0.8;
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
B = get_fem_matrix(mesh,'B_p1dg0');

%P1 mass matrix for error computation
M_p1 = get_fem_matrix(mesh,'mass_p1');
R = chol(tau*A + M_p1);
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
    L = tau*L1 + A*u_old - tau*L2;
    u = zeros(mesh.nc,1);
    y = R_T\L;
    y1 = R\y;
    u = y1;
    
    MSE = norm(u_true-u)^2/512^2;
    psnr = 10*log10(1/MSE);
    psnrvec = [psnrvec;psnr];
     
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


