import sys
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time

par = Parameters()
par.add("beta", 0.8)
par.parse(sys.argv)
    
# Regularization parameters
beta = par["beta"]
mue = 1.

# Algorithm parameters
MaxIter = 500
rho = 1.0

mesh = UnitSquareMesh(128,128)

f = Expression("10.*sin(x[0]*3*3.1416)*cos(x[1]*2*3.1416)*(2-0.5*x[0]*x[1])", degree=2)

def boundary(x, onboundary):
    return onboundary #x[0]<DOLFIN_EPS
    
file = File("velocity.pvd")

Y = FunctionSpace(mesh, "Lagrange", 1)
W = VectorFunctionSpace(mesh, "DG", 0)

bc = DirichletBC(Y, Constant("0.0"), boundary)

y = TrialFunction(Y)
v = TestFunction(Y)
w = TestFunction(W)

y_old = Function(Y)
y_cur = interpolate(Constant("0.0"), Y)
d_cur = interpolate(Expression(("0.0","0.0"), degree=1), W)
lam = interpolate(Expression(("0.0","0.0"), degree=1), W)

a = (mue+rho)*inner(grad(y),grad(v))*dx
L = f*v*dx + rho*inner(d_cur-lam,grad(v))*dx
    
dist_list = []
err_list = []

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

    y_old.assign(y_cur);

    # Step 1: Minimize for y
    t = time.time()

    solve(a == L, y_cur, bc)

    elapsed = time.time() - t
    print("Solution of Poisson took ", elapsed, "seconds")
    
    # Step 2: Shrinkage
    d_vec = d_cur.vector()

    grad_y = project(grad(y_cur), W).vector()
    lam_vec = lam.vector()

    t = time.time()

    n=len(lam_vec)
    z=(grad_y+lam_vec).get_local().reshape(int(n/2),2)
    
    z_norm = np.linalg.norm(z,axis=1)
    
    d_vec=np.zeros_like(z)
    idx = z_norm>beta/rho

    d_vec[idx,0] = (z_norm[idx] - beta/rho) * z[idx,0]/z_norm[idx]
    d_vec[idx,1] = (z_norm[idx] - beta/rho) * z[idx,1]/z_norm[idx]
    
    d_vec = d_vec.reshape((n,))
    d_cur.vector()[:] = d_vec

    elapsed = time.time() - t
    print("Shrinkage took ", elapsed, "seconds")
    
    # Step 3: Multiplier update  
    t = time.time()

    lam_vec = lam_vec + rho*(grad_y - d_vec)
    lam.vector()[:] = lam_vec

    elapsed = time.time() - t
    print("Multiplier update took ", elapsed, "seconds")
        
    dist = sqrt(assemble((y_cur - y_old) ** 2 * dx))
    print(dist)
    dist_list.append(dist)

    y_cur.rename("y", "y")
    file << y_cur,k

    # Save current iterate
    y_all.append(Function(Y))
    y_all[k].vector()[:] = y_cur.vector()
    
    k = k+1

MaxIter = k    

for k in range(MaxIter):
    err = sqrt(assemble((y_all[k]-y_all[MaxIter-1])**2*dx))
    err_list.append(err)
    
np.savetxt("err_l2.csv", err_list, delimiter="\n")
np.savetxt("dist_l2.csv", dist_list, delimiter="\n")

    
    
    
