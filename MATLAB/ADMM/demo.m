%DEMO
%ADMM method for Y = L2
[u_vec,d_vec,lamb_vec,distv] = admm_l2(32,@fun,50,7.0);

%Alternatively:

%Y = H_curl, bestapproximation of shrinkage approach
%[u_vec,d_vec,lamb_vec,distv] = admm_l2_ned(32,@fun,50,7.0);

%Y = H_curl, regularization approach
%[u_vec,d_vec,lamb_vec,distv] = admm_curl(32,@fun,50,7.0);

%Chambolle and Pocks method
%[u_vec,d_vec,u_bar_vec,distv] = cham_pock(32,@fun,50);

mesh = create_unitsquaremesh(32);

%plot mesh
plot_mesh(mesh);
title('mesh');

%plot solution u_h after 5 iterations
fg2 = figure;
plot_function(mesh,u_vec(:,:,5));
title('5 iterations');

%plot solution u_h after 50 iterations
fg3 = figure;
plot_function(mesh,u_vec(:,:,50));
title('50 iterations');

%plot the distances ||u_h,k+1 - u_h,k||_L^2
fg4 = figure;
semilogy(distv);
title('||u_{h,k+1} - u_{h,k}||_{L^2}');