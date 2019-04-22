import numpy as np
from dolfin import *
import csv

def max(a,b): return 0.5*(a+b+abs(a-b))

def shrink_l2(a,b):
    norm_a = sqrt(inner(a,a))
    return a/norm_a*max(norm_a-b,0)

def shrink_l1(a,b):
    norm_a = abs(a[0])+abs(a[1])
    return a/norm_a*max(norm_a-b,0)


class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS



# Create mesh and define function space
mesh = UnitSquareMesh(64,64)
mesh1 = UnitSquareMesh(256,256)
V1 = FunctionSpace(mesh1, "Lagrange", 1)
V = FunctionSpace(mesh, "Lagrange", 1)
#W = FunctionSpace(mesh,"N1curl",1)
#DG = FunctionSpace(mesh,"DG",0)
W = VectorFunctionSpace(mesh,"DG",0)

#Read solution
u_sol1 = Function(V1)
hdf = HDF5File(mesh.mpi_comm(), "uf_256_DG0.h5", "r")
hdf.read(u_sol1, "/cf1")
hdf.close()

u_sol = Function(V)
u_sol = project(u_sol1,V)

file1 = XDMFFile('u_l2.xdmf')
file2 = XDMFFile('d_l2.xdmf')
file3 = XDMFFile('lamb_32.xdmf')

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, Boundary())

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
u_old = Function(V)
lamb = Function(W)
d = Function(W)
d0 = Constant([0.0001,0.0001])
d = interpolate(d0,W)
lamb = interpolate(d0,W)

# = Expression("10.*sin(x[0]*3*3.1416)*cos(x[1]*2*3.1416)*(2-0.5*x[0]*x[1])", degree=2)
f = Expression("5.0+5*sin(10*x[0])+5*sin(10*x[1])", degree=2)


mu = 1.0
gamma = 7.0
beta = 0.5
tol = 1e-14
dist = 10.0
it = 200
k=0
errv = []
err = 10

#Define linear variational problem
a = (mu+gamma)*inner(grad(u), grad(v))*dx
L = f*v*dx+inner(gamma*d+lamb, grad(v))*dx

u = Function(V)

while (dist>tol and k<it):

    solve(a == L, u, bc)
    d1 = shrink_l2(grad(u) - lamb / gamma, beta/gamma)
    d.assign(project(d1,W))
    print(sqrt(assemble(inner(d, d) * dx)))
    l_n = lamb - gamma * (grad(u) - d)
    lamb.assign(project(l_n, W))
    dist = sqrt(assemble((u - u_old) ** 2 * dx))
    print(dist)
    #err = sqrt(assemble((u - u_sol) ** 2 * dx))
    #print('err =',  err)
    u_old.assign(u)
    k += 1
    file1.write(u, float(k))
    file2.write(d, float(k))
    file3.write(u_sol1)
    print(k)
    #errv.append(err)


# Save error
with open('err.txt',  'w', newline='') as f:
    wr = csv.writer(f, delimiter = '\n',quoting=csv.QUOTE_MINIMAL)
    wr.writerow(errv)

#hdf = HDF5File(mesh.mpi_comm(), "file_u64.h5", "w")
#hdf.write(u, "/cf1")
#hdf.close()

#hdf = HDF5File(mesh.mpi_comm(), "file_d.h5", "w")
#hdf.write(d, "/cf2")
#hdf.close()

#hdf = HDF5File(mesh.mpi_comm(), "file_lamb.h5", "w")
#hdf.write(lamb, "/cf3")
#hdf.close()
