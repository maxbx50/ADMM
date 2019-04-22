function[mesh] = create_unitsquaremesh(N)
%CREATE_UNITSQUAREMESH
%   Creates a conforming trianguluation of the unit square [0,1]x[0,1] with 
%   N cells in x and y direction and bisecting each one of them.
%   The vertices are numbered in lexicographic order.
%
%   INPUT:     N     -     number of cells in x and y direction
%
%   Example for N=2
%
%   mesh.coordinates = [0 0.5 1 1 0.5 0 0 0.5 1; 0 0 0 0.5 0.5 0.5 1 1 1];
%   mesh.elements = [1 2 4; 2 5 4; 2 3 5; 3 6 5; 4 5 7; 5 8 7; 5 6 8; 6 9 8];
%

fprintf('Creating mesh...\n');

timer = tic;

%mesh points
mesh.ncells = N;
mesh.nc = (N+1)^2;
mesh.coordinates = zeros(2,mesh.nc);
x = 0:1/N:1;
y = 0:1/N:1;
[xx,yy]=meshgrid(x,y);
x = reshape(xx',[mesh.nc,1]);
y = reshape(yy',[mesh.nc,1]);
mesh.coordinates(1,:) = x;
mesh.coordinates(2,:) = y;

mesh.dim = size(mesh.coordinates,1); 

%mesh elements specified by their vertex numbers
temp1 = [1 2 N+2];
temp2 = [2 N+3 N+2];
j=1:2:2*N^2;
r = (j'-1)/2 + ceil(j' / (2*N))-1;
mesh.elements = zeros(2*N^2,size(temp1,2));
mesh.elements(1:2:end,:) = repmat(temp1,N^2,1)+r;
mesh.elements(2:2:end,:) = repmat(temp2,N^2,1)+r;

mesh.nel = size(mesh.elements,1);

%collect all edges for each element
edges = zeros(3*mesh.nel,2);
edges(1:3:end,:) = mesh.elements(:,[1,2]);
edges(2:3:end,:) = mesh.elements(:,[2,3]);
edges(3:3:end,:) = mesh.elements(:,[3,1]);

%delete double counted edges
edges_sorted = sort(edges,2);
[~,ind_edges,ind_nodes] = unique(edges_sorted,'rows');
mesh.edges_vertices = edges(ind_edges,:);
mesh.ned = size(mesh.edges_vertices,1);
mesh.edges_elements = reshape(ind_nodes,3,mesh.nel)';

%identify boundary edges
bd_edges1 = [1,2]+(0:1:N-1)';
bd_edges2 = [N+1,2*(N+1)]+(N+1)*(0:1:N-1)';
bd_edges3 = [N*(N+1)+2,N*(N+1)+1]+(0:1:N-1)';
bd_edges4 = [N+2,1]+(N+1)*(0:1:N-1)';
mesh.bd_edges = [bd_edges1;bd_edges2;bd_edges3;bd_edges4];

%set signs for all the edges e_ij which is +1 if i<j and -1 otherwise
signs = mesh.elements(:,[2 3 1]) - mesh.elements(:,[1 2 3]);
mesh.signs = signs ./ abs(signs);

%get the affine transformations for each element
[mesh.B_K, mesh.b_K, mesh.detB_K, mesh.B_K_invT] = affine_transformation(mesh);

time = toc(timer);

fprintf('Finished creating mesh. Time elapsed: %.4f seconds. \n', time);


end

