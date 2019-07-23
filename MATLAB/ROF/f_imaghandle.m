function out = f_imaghandle(x,noisy_image,shape)
%F_IMAGHANDLE
%   function handle for an (noisy) image
%
%   INPUT:   x            -        matrix containing evaluation points
%            noisy_image  -        noisy image as matrix
%            shape        -        image shape
%
%   OUTPUT:  out       -        tensor containing the iterates u_k

ind_x = max(shape(1) - ceil(x(:,1) * shape(1)), 1);
ind_y = max( ceil(x(:,2) * shape(2)), 1);

linearidx = sub2ind(size(noisy_image),ind_x,ind_y);

noise_image = noisy_image(:);


out =  noisy_image(linearidx); 

end
