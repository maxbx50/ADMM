%DEMO
%ADMM method for Y = L2, Y_h = DG0
N=256;

%[u_vec,psnrvec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = ROF_l2(N,50,0.0001,0.015/120,'Lena.png');

%Alternatively:

%Y = H_curl, bestapproximation of shrinkage approach
%[u_vec,psnrvec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = ROF_l2_ned(N,50,0.0001,0.015/120,'Lena.png');

%Y = H_curl, regularization approach
%[u_vec,psnrvec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = ROF_curl(N,50,0.0001,0.015/120,'Lena.png');

%plot mesh
%plot_mesh(mesh);
%title('mesh');

%plot the psnr 
if(N==512)
    fg1 = figure;
    plot(psnrvec);
    title('PSNR');
    xlabel('Iterations');
end

%plot the distances ||u_h,k+1 - u_h,k||_L^2
fg2 = figure;
semilogy(distv); hold on; semilogy(r_pvec); semilogy(r_dvec);
xlabel('Iterations');
legend('||u_{h,k+1} - u_{h,k}||_{L^2}','primal residual','dual residual');