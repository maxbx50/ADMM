import numpy as np
from dolfin import *
import csv
import matplotlib.pyplot as plt

def shrink(a,b):
    abs_a = np.absolute(a)
    ind = abs_a > b
    out = np.zeros(len(a))
    out[ind] = a[ind]/abs_a[ind]*(abs_a[ind]-b[ind])
    return out


class Interior(SubDomain):
    def inside(self, x, on_boundary):
        return not on_boundary

file1 = XDMFFile('u.xdmf')
file2 = XDMFFile('d.xdmf')
file3 = XDMFFile('lamb.xdmf')

# Create mesh and define function space
#mesh = UnitSquareMesh(2, 2, "crossed")
mesh = UnitSquareMesh(512, 512, "crossed")

W = FunctionSpace(mesh,"Discontinuous Lagrange Trace",0)
DG = FunctionSpace(mesh,"DG",0)
DGV = VectorFunctionSpace(mesh,"DG",0)
dofmap = W.dofmap()


#find dof mapping for inner facets
inner_dofs = []
facet_length = []
midpoints_facets_p = []
midpoints_facets_n = []
epsi = 1e-7

for cell in cells(mesh):
    dofs = dofmap.cell_dofs(cell.index())
    # dofs_x = dofmap.tabulate_coordinates(cell)
    for i, facet in enumerate(facets(cell)):
        p = facet.midpoint()
        if not (near(p.x(), 0.0) or near(p.x(), 1.0)) and not (near(p.y(), 0.0) or near(p.y(), 1.0)):
            midpoints_facets_p1 = [p.x() + epsi * facet.normal().x(), p.y() + epsi * facet.normal().y()]
            midpoints_facets_n1 = [p.x() - epsi * facet.normal().x(), p.y() - epsi * facet.normal().y()]

            midpoints_facets_p.append(midpoints_facets_p1)
            midpoints_facets_n.append(midpoints_facets_n1)
            inner_dofs.append(dofs[i])
            facet_length.append(cell.facet_area(i))

midpoints_facets_p = np.array(midpoints_facets_p)
midpoints_facets_n = np.array(midpoints_facets_n)
facet_length = np.array(facet_length)
inner_dofs = np.transpose(inner_dofs)
n_ed = len(midpoints_facets_p[:,0])

#load image
image = np.loadtxt('lena.dat', delimiter=' ')
image = np.transpose(image)

#generate noisy image
np.random.seed(seed=1)
noise_std_dev = 20
noise = noise_std_dev*np.random.randn(image.shape[0], image.shape[1])
noisy_image = image + noise
noisy_image = noisy_image.flatten()
true_image = image.flatten()

#transform images into DG0 functions
image_true_function = Function(DG)
image_noisy_function = Function(DG)

im_tv = np.zeros_like(image_true_function.vector()[:])
im_nv = np.zeros_like(image_true_function.vector()[:])

im_tv[0::4] = true_image
im_tv[1::4] = true_image
im_tv[2::4] = true_image
im_tv[3::4] = true_image

im_nv[0::4] = noisy_image
im_nv[1::4] = noisy_image
im_nv[2::4] = noisy_image
im_nv[3::4] = noisy_image

image_true_function.vector()[:] = im_tv
image_noisy_function.vector()[:] = im_nv


#file1.write(image_true_function)
file2.write(image_noisy_function)


# Define variational problem
u = TrialFunction(DG)
v = TestFunction(DG)
#u_old = interpolate(image_noisy_function,DG)
u_old = interpolate(Constant(0.0),DG)
d = Function(W)
lamb = Function(W)


mu = 1.0
gamma = 7.0
beta = 1.0
tol = 1e-14
dist = 1.0
it = 3
k=0
distv = []
errv = []
err = 1.0
n = FacetNormal(mesh)


#Define linear variational problem
a_1 = u*v*dx
#a_2 = inner(grad(u),grad(v))*dx - dot(avg(grad(v)), jump(u, n))*dS + dot(jump(v, n), avg(grad(u)))*dS - dot(grad(v), u*n)*ds + dot(v*n, grad(u))*ds
a_2 = inner(grad(u),grad(v))*dx - dot(avg(grad(v)), jump(u, n))*dS - dot(jump(v, n), avg(grad(u)))*dS - dot(grad(v), u*n)*ds - dot(v*n, grad(u))*ds
a = mu*a_1 + gamma*a_2
L = mu*image_noisy_function*v*dx + inner(jump(v,n), avg(gamma*d*n+lamb*n))*dS + inner(jump(gamma*d+lamb,n), avg(v*n))*dS


u = Function(DG)


while (dist > tol and k<it):
    k += 1
    print('Iteration:',k)

    #u-problem
    solve(a == L, u)

    #d-problem
    u_j = np.zeros_like(d.vector()[inner_dofs])
    for i in range(0,n_ed):
        u_j[i] = u(midpoints_facets_p[i]) - u(midpoints_facets_n[i])

    #u_j = interpolate(jump(u),W)

    d.vector()[inner_dofs] = shrink(u_j - lamb.vector()[inner_dofs] / gamma, facet_length*beta / gamma)
    #print(np.linalg.norm(d.vector()[:]))
    lamb.vector()[inner_dofs] -= gamma * (u_j - d.vector()[inner_dofs])
    #print(np.linalg.norm(lamb.vector()[:]))

    #Convergence test
    dist = sqrt(assemble((u - u_old) ** 2 * dx))
    print('Distance:',dist)

    #update and save data
    u_old.assign(u)
    file1.write(u, float(k))
    distv.append(dist)


# Save error
with open('dist.txt',  'w', newline='') as f:
    wr = csv.writer(f, delimiter = '\n',quoting=csv.QUOTE_MINIMAL)
    wr.writerow(distv)
