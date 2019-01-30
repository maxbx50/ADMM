import sys
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

par = Parameters()
par.add("beta", 0.8)
par.parse(sys.argv)
    
# Regularization parameters
beta = par["beta"]
mue = 1.
eps = 0.000001

# Algorithm parameters
MaxIter = 500
rho = 1.0

mesh = UnitSquareMesh(128,128)

f = Expression("10.*sin(x[0]*3*3.1416)*cos(x[1]*2*3.1416)*(2-0.5*x[0]*x[1])", degree=2)

def boundary(x, onboundary):
    return onboundary #x[0]<DOLFIN_EPS
    
file = File("velocity.pvd")

Y = FunctionSpace(mesh, "Lagrange", 1)
NE = FiniteElement('N1curl', triangle, 1)
W = FunctionSpace(mesh, NE)

bc_Y = DirichletBC(Y, Constant(0.0), boundary)
bc_W = DirichletBC(W, Constant((0.0,0.0)), boundary)

y_trial = TrialFunction(Y)
v = TestFunction(Y)
w = TestFunction(W)

y_old = Function(Y)
y = interpolate(Constant("0.0"), Y)
d = interpolate(Expression(("0.0","0.0"), degree=1), W)
lam = interpolate(Expression(("0.0","0.0"), degree=1), W)

a = (mue+rho)*inner(grad(y_trial),grad(v))*dx
L = f*v*dx + rho*inner(d+lam,grad(v))*dx + rho*inner(curl(d+lam),curl(grad(v)))*dx

dist_list = []
err_list = []
eps_list = []

y_all = []

print('beta = ', beta)

dist = 1.0
k = 0

while k < MaxIter and dist > 1.e-11:

    print('--------------------------------------------------')
    print('    Iteration: ', k)
    print('--------------------------------------------------')

    # ADMM method
    #######################

    y_old.assign(y);

    # Step 1: Minimize for y
    solve(a == L, y, bc_Y)

    # Step 2: Shrinkage
    F = rho*inner(d-grad(y)+lam,w)*dx + rho*inner(curl(d-grad(y)+lam),curl(w))*dx \
    + beta*inner(d/sqrt(inner(d,d)+eps), w)*dx
    
    solve(F == 0, d, bc_W);    
    
    # Step 3: Multiplier update  
    lam.assign(lam + rho*(project(d-grad(y),W)))
    
    dist = sqrt(assemble((y - y_old) ** 2 * dx))
    print("Distance to previous iterate: ", dist)
    
    dist_list.append(dist)
    eps_list.append(eps)
    
    if dist<1.e-4 and eps > 1.e-12:
        print("DESCREASING EPS:", eps)
        eps = eps/2
    
    y.rename("y", "y")
    file << y,k

    # Save current iterate
    y_all.append(Function(Y))
    y_all[k].vector()[:] = y.vector()
    
    k = k+1

MaxIter = k

for k in range(MaxIter):
    err = sqrt(assemble((y_all[k]-y_all[MaxIter-1])**2*dx))
    err_list.append(err)
    
np.savetxt("err_curl.csv", err_list, delimiter="\n")
np.savetxt("dist_curl.csv", dist_list, delimiter="\n")
np.savetxt("eps_curl.csv", eps_list, delimiter="\n")

    
    
    
