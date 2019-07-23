import sys
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

par = Parameters()
par.add("beta", 1.e-5)
par.parse(sys.argv)
    
# Regularization parameters
beta = par["beta"]
gamma = 1.e-6
mue = 1.e-8

# Model parameters
noise_strength = 0.05
person = "Stoetzi"

# Parameters for adaptivity
MaxAdaptIter = 10
refinement_threshold = 0.9

# Parameters for the solver
MaxNewtonIter = 15

mesh = UnitSquareMesh(8,8)

# f = Expression("10.*sin(x[0]*2*3.1416)*cos(x[1]*2*3.1416)*(2-0.5*x[0]*x[1])", degree=2)

def boundary(x, onboundary):
    return False #onboundary #x[0]<DOLFIN_EPS

im = Image.open(person + '.png', 'r')
print(im.format, im.size, im.mode)

width, height = im.size
pixel_values = im.load()

orig=[[0 for row in range(height)] for col in range(width)]
noisy=[[0 for row in range(height)] for col in range(width)]
noisy_data=[[0 for row in range(height)] for col in range(width)]

for i in range(width):
    for j in range(height):
        pixel = pixel_values[i,j]
        if im.mode == 'RGB':
            orig[i][j] = (pixel[0] + pixel[1] + pixel[2])/(3*255)
        elif im.mode == 'L':
            orig[i][j] = pixel/(255)

        noisy[i][j] = orig[i][j] # + np.random.normal(0,noise_strength)
        noisy_data[i][j] = int(min(254,max(0,3*255*noisy[i][j])))
        noisy[i][j] = 1-noisy[i][j]
        
im_noisy = Image.new('L',(height,width))
im_noisy.putdata(sum(noisy_data, []))
im_noisy.rotate(90)
#im_noisy.show()
im_noisy.save('Noisy' + person + '.png')

class SourceTerm(Expression):
    def eval(self, value, x):
        if False:
            if 0.2**2. < (x[0]-0.5)**2 + (x[1]-0.5)**2 < 0.4**2:
                value[0] = 1.0
            else:
                value[0] = 0.0
        else:
            value[0] = noisy[min(int(x[0]*width),width-1)][height-1- min(int(x[1]*height),height-1)]
        
f = SourceTerm(degree=2)    

file = File("sol_vi_adapt.pvd")
file_p = File("sol_vi_adapt_p.pvd")

for k in range(MaxAdaptIter):

    print('--------------------------------------------------')
    print('    Iteration: ', k)
    print('--------------------------------------------------')

    
    V_el = FiniteElement("Lagrange", triangle, 1)
    P_el = VectorElement("DG", triangle, 0)
    VP_el = MixedElement([V_el,P_el])
    VP = FunctionSpace(mesh, VP_el)

    up = Function(VP)
    dup = TrialFunction(VP)
    
    u, p = split(up)
    du, dp = split(dup)
    vt = TestFunction(VP)
    v, t = split(vt)

    u0 = Constant(0.0)
    bc = DirichletBC(VP.sub(0), u0, boundary)

    F = mue*inner(grad(u),grad(v))*dx + u*v*dx \
        + inner(p,grad(v))*dx \
        - f*v*dx \
        - beta*inner(grad(u),t)*dx \
        + Max(gamma,(dot(grad(u),grad(u))+0.01*gamma)**0.5)*inner(p,t)*dx

    # Works!
    '''
    solve(F==0, up, bc)
    res = assemble(F)
    bch.apply(res)
    resid = norm(res)
    print("Residuum=", np.max(np.abs(res.array())))
    sys.exit()
    '''
    
    J = derivative(F, up)

    '''
    # Newton direction
    dup = Function(VP)
    solve(J == F, dup, bch)
    up.vector()[:] = up.vector() - dup.vector()
    res = assemble(F)
    bch.apply(res)
    resid = norm(res)    
    print("Residuum=", np.max(np.abs(res.array())))
    sys.exit()
    '''
    
    # Setup initial guess
    #u_init = interpolate(Expression("x[0] + x[1]", degree=2), VP.sub(0).collapse())
    u_init = interpolate(f, VP.sub(0).collapse())
    p_init = interpolate(Expression(("0.0","0.0"), degree=2), VP.sub(1).collapse())
    assign(up, [u_init,p_init])

    # plot(u_init)
    # plt.show()

    res = assemble(F)
    # bc.apply(res)
    resid_old = norm(res)
    
    for i in range(MaxNewtonIter):

        if resid_old < 1.e-7:
            break

        # Newton direction
        dup = Function(VP)
        solve(J == F, dup) #, bc)
   
        up_cur = Function.copy(up, deepcopy=True)
        print("Start Residuum=", resid_old)

        alpha = 1.0
        while True:
            up.vector()[:] = up_cur.vector() - alpha*dup.vector()
            res = assemble(F)
            #bch.apply(res)
            resid = norm(res)
            print("Residuum=", resid, ", alpha=", alpha)

            if resid <= resid_old: # or alpha < 10.e-5:
                break
            
            alpha = 0.5*alpha

        resid_old = resid
                
    # problem = NonlinearVariationalProblem(F, up, bc, J);
    # solver = NonlinearVariationalSolver(problem)
    # solver.solve()
    
    u, p = up.split()

    u.rename("u", "u")
    file << u,k
    p.rename("p", "p")
    file_p << p,k

    # sys.exit(0)
    
    # Adaptive refinement
    cell_markers = MeshFunctionBool(mesh, 2)
    cell_markers.set_all(False)
    
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)

    DG0 = FunctionSpace(mesh, "DG", 0)
    w= TestFunction(DG0)
    
    eta_T = assemble(h**2*(f-u)**2*w*dx)
    eta_dT_u = assemble(avg(h)*jump(grad(u),n)**2*avg(w)*dS)
    eta_dT_p = assemble(avg(h)*jump(p,n)**2*avg(w)*Constant(0.0)*dS)
    eta_func = Function(DG0)
    # eta_func.vector()[:] = 0.5*(eta_dT_u.array() + eta_dT_p.array())
    eta_func.vector()[:] = eta_T.get_local() + 0.5*(eta_dT_u.get_local() + eta_dT_p.get_local())
    eta = eta_func.vector().get_local()
    
    # Sort descending
    idx = np.argsort(-eta)

    # Sum of eta
    ref_idx = idx[ np.cumsum(eta[idx]) < refinement_threshold * sum(eta) ]
    cell_markers.array()[ref_idx] = True
    
    print("Refining " + str(len(ref_idx)) + " of " + str(len(eta)) + " elements.")
    
    mesh = refine(mesh, cell_markers)

    
    
    
