function [errv] = calculate_error(M,u_vec,maxit)
errv = [];

for k = 1:200
    err = sqrt((u_vec(:,:,k)-u_vec(:,:,maxit))'*M*(u_vec(:,:,k)-u_vec(:,:,maxit)));
    fprintf('\nerror ||u_k,h - u_N,h||_L^2 : %.7e \n', err);
    errv = [errv;err];
end

end

