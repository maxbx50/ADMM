%DEMO
%ADMM method for Y = L2, Y_h = DG_0
[u_vec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = admm_l2(128,@fun,200,7.0);

%Alternatively:

%Y = L_2, Y_h = N_0
%[u_vec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = admm_l2_ned(128,@fun,200,7.0);

%Y = H_curl, Y_h = N_0
%[u_vec,d_vec,lamb_vec,distv,r_pvec,r_dvec,gammav,mesh] = admm_curl(128,@fun,200,7.0);

%Chambolle and Pocks method
%[u_vec,d_vec,u_bar_vec,distv,mesh] = cham_pock(128,@fun,200);

%plot mesh
%plot_mesh(mesh);
%title('mesh');

%plot solution u_h after 5 iterations
fg2 = figure;
plot_function(mesh,u_vec(:,:,5));
title('5 iterations');
xlabel('x');
ylabel('y');

%plot solution u_h after 50 iterations
fg3 = figure;
plot_function(mesh,u_vec(:,:,50));
title('50 iterations');
xlabel('x');
ylabel('y');

%plot the distances ||u_h,k+1 - u_h,k||_L^2
fg4 = figure;
semilogy(distv); hold on; semilogy(r_pvec); semilogy(r_dvec);
xlabel('Iterations');
legend('||u_{h,k+1} - u_{h,k}||_{L^2}','primal residual','dual residual');
