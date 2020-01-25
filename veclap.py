from dolfin import *
# Create mesh
mesh = UnitSquareMesh(10, 10)
# Define Exact solution
#u_D = Expression(('0.5*pow(x[0],2)','pow(x[1],2)'),degree=2)
u_D = Expression(('0.5*(x[0] - pow(x[0], 2))','x[1] - pow(x[1], 2)'),degree=2)
# Define function spaces
NED = FiniteElement("N1curl", mesh.ufl_cell(), 2)
#RT = FiniteElement("RT",  mesh.ufl_cell(), 1)
CG = FiniteElement("CG", mesh.ufl_cell(), 1)
BE = VectorElement("Bubble",mesh.ufl_cell(),3)
#W = CG * NED
#W = CG * RT
#V = FunctionSpace(mesh,W)
# Define trial and test functions
#(sigma, u) = TrialFunctions(V)
#(tau, v) = TestFunctions(V)
# Define bilinear form and right hand side
def a(sigma, u, tau, v):
    return (sigma*tau - inner(u, grad(tau))+ inner(grad(sigma), v) + inner(curl(u),curl(v)))*dx
def L(v):
    f = Constant((-1., -2.))
    return dot(f, v)*dx
#a = (sigma*tau - dot(u, grad(tau))+ dot(grad(sigma), v) + inner(curl(u),curl(v)))*dx
#f = Constant((-1., -2.))
#L = dot(f, v)*dx
# solve and plot
V_mixed = FunctionSpace(mesh,MixedElement([CG, NED]))
#w = TrialFunction(V_mixed)
(sigma, u) = TrialFunctions(V_mixed)
#dw = TestFunction(V_mixed)
(tau, v) = TestFunctions(V_mixed)

sigma_enriched = sigma[0] + sigma[1]
u_enriched = u[0] + u[1]

tau_enriched = tau[0] + tau[1]
v_enriched = v[0] + v[1]
wh = Function(V_mixed)
#w = Function(V)
solve(a(sigma_enriched, u_enriched, tau_enriched, v_enriched)==L(v_enriched),wh)
uh_enriched = wh[0] + wh[1]
#solve(a == L, w)
sigma, u = wh.split()
#plot(u)
#plot(sigma)
#import matplotlib.pyplot as plt
#plt.show()
# Save solution to file in VTK format
#vtkfile = File('veclapNED2/solution.pvd')
#vtkfile << u
# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')
# Print errors
print('error_L2  =', error_L2)
#Hold plot
#interactive()
