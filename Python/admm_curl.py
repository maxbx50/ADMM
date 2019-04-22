import numpy as np
from dolfin import *
import csv

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
W = FunctionSpace(mesh,"N1curl",1)
DG = FunctionSpace(mesh,"DG",0)
#W = VectorFunctionSpace(mesh,"DG",0)


#Read solution on refined mesh for error estimations
#u_sol1 = Function(V1)
#hdf = HDF5File(mesh.mpi_comm(), "u_256.h5", "r")
#hdf.read(u_sol1, "/cf1")
#hdf.close()
#u_sol = Function(V)
#u_sol = project(u_sol1,V)


file1 = XDMFFile('test.xdmf')
file2 = XDMFFile('test1.xdmf')
file3 = XDMFFile('lamb.xdmf')


# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, Boundary())


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
w = TestFunction(W)
w1= TestFunction(W)
u_old = Function(V)
lamb = Function(W)
d = Function(W)
d1 = Constant([1.0,1.0])
d0 = Constant([0.0,0.0])
d = interpolate(d0,W)
d_s = interpolate(d0,W)
lamb = interpolate(d0,W)
d_old = interpolate(d0,W)


#Define problem data and parameters
#f = Expression("10.*sin(x[0]*3*3.1416)*cos(x[1]*2*3.1416)*(2-0.5*x[0]*x[1])", degree=2)
f = Expression("5.0+5*sin(10*x[0])+5*sin(10*x[1])", degree=2)

mu = 1.0
gamma = 7.0
beta = 0.5
tol = 1e-15
dist = 10.0
eps1 = 1e-6


#Define forms and further parameters
a = (mu+gamma)*inner(grad(u), grad(v))*dx
L = f*v*dx+inner(gamma*d+lamb, grad(v))*dx

u = Function(V)
it = 200
k=0
errv = []
err=10
dist_old=10


while (err>tol and k<it):

    solve(a == L, u, bc)
    grad_p = project(grad(u), W)
    F = gamma*inner(d-grad_p+lamb/gamma,w)*dx+gamma*inner(curl(d+lamb/gamma),curl(w))*dx+(beta*inner(d,w)/sqrt(inner(d,d)+eps1))*dx
    solve(F == 0, d, J=derivative(F, d), solver_parameters={"newton_solver": {"relative_tolerance": 1e-10},
                                                            "newton_solver": {"maximum_iterations": 100}})
    print(sqrt(assemble(inner(d,d)*dx)))
    l_n = lamb - gamma*(grad(u)-d)
    lamb.assign(project(l_n,W))
    dist = sqrt(assemble((u-u_old)**2*dx))
    print(dist)
    #err = sqrt(assemble((u - u_sol) ** 2 * dx))
    #print('err =', err)
    u_old.assign(u)
    d_old.assign(d)
    k += 1
    #dist_old=dist
    file1.write(u_old,float(k))
    file2.write(d_old,float(k))
    file3.write(lamb,float(k))
    #errv.append(err)
    print(k)


#Save error
with open('err_c.txt',  'w', newline='') as f:
    wr = csv.writer(f, delimiter = '\n',quoting=csv.QUOTE_MINIMAL)
    wr.writerow(errv)
