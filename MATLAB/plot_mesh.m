function plot_mesh(mesh)
%PLOT_MESH
%   Function to plot the unit square mesh
%
%   INPUT:   mesh       -        mesh as a structure

x = reshape(mesh.coordinates(1,mesh.elements'),3,mesh.nel);
y = reshape(mesh.coordinates(2,mesh.elements'),3,mesh.nel);

fg = figure;

col1 = [136/255,181/255,1];

fill(x,y,col1);

xlabel('x');
ylabel('y');

end


