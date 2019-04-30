function [] = plot_function(mesh,u_h)
%PLOT_FUNCTION
%   Plots the scalar grid function u_h
%
%   INPUT:   mesh      -        mesh as a structure
%            u_h       -        grid function

N = mesh.ncells;
x = 0:1/N:1;
y = 0:1/N:1;
[xx,yy]=meshgrid(x,y);

u_h = reshape(u_h,N+1,N+1);

surf(xx,yy,u_h);
colormap winter
xlabel('x');
ylabel('y');

view(-70,20);

end

