""" Arzhang Angoshtari, Ali Gerami, March 2020
This code estimates the lower bound $\beta_h$ for the material-dependent inf-sup condition by 
using the smallest eigenvalue of a suitable matrix 
Solving 2D nonlinear elasticity problem using the three-field formulation: 
Find (U,K,P)\in [H^{1}]^{n}*H^{c}*H^{d} such that
            <P, grad(Ups)> =  <B, Ups> + int_{\Gamma_2}\bar{T}. Ups ds  forall Ups \in H^{1}_1
 <grad(U), Lam> - <K, Lam> = 0                                          forall Lam \in H^{c}
   <Pbb(K), Pi > - <P, Pi> = 0                                          forall Pi  \in H^{d}
The output is the convergence rates.
The Newton method (on the PDE level) is employed. The exact solution is
U_e = [0.5*y**3 + 0.5*Coef_e*sin(0.5*pi*y)]. BC condition is imposed as Dirichelt BC for 
displacement, i.e. \Gamma_2 = \emptyset

Input parameters:
    degreeCG, degreeCurl, degreeDiv       degrees of CG, curl and div finite elements
    Types of CG, curl, and div elements
    an array containing n (mesh sizes)
    Guass_points: number of points for Guassian Quadrature, enter 0 to use FEniCS default
    omega:  relaxation parameter, omega<1 may improve convergence in some problems
    tol: convergence tolerance for the infinity norm of the increment of Newton's method
    maxiter: maximum allowed iteration for Newton's method
    normtype in errornorm
    degRise: degree_rise in errornorm
    mu, lam: Parameters of Neo-Hookean materials
    Coef_e: If set to zero, the exact solution will be polynomial, and errors very small using suitable degree for elements
Output:
    $\beta_h$ and convergence rates
"""

from dolfin import *
from mshr import *

import numpy as np
import matplotlib.pyplot as plt

def compute(n, degreeCG, degreeCurl, degreeDiv):
    """ Given mesh division size n and the degree of elements (degreeCG, degreeCurl, degreeDiv),
    this function return the error of solving the above mixed formulation of nonlinear elasticity
    with respect to the exact solution U_e"""
    
    # Create mesh and define function space
    
    #       Structured mesh
    #mesh = UnitSquareMesh(n, n)
        
    #       Unstructured mesh
    Rec = Rectangle(Point(0, 0), Point(1, 1))
    mesh = generate_mesh(Rec, n)
    
    CGE  = FiniteElement("CG", mesh.ufl_cell(), degreeCG)
    CurlE = FiniteElement("N1curl", mesh.ufl_cell(), degreeCurl)  # N1curl and N2curl
    DivE = FiniteElement("RT", mesh.ufl_cell(), degreeDiv)        # RT and BDM  
    Z = FunctionSpace(mesh, MixedElement([CGE, CGE, CurlE, CurlE, DivE, DivE]))
    
    # Define boundary conditions
    def Ue_boundary(x, on_boundary):
        return on_boundary                         #the whole boundary    
    
    #def Ue_boundary(x, on_boundary):
    #    tol = 1E-12   # tolerance for coordinate comparisons
    #    return on_boundary and abs(x[0]) < tol and abs(x[1]) < tol    # only the origin
                                  
    
    #       Homogeneous BCs for the increment in each Newton's iteration
    bcs_h = [DirichletBC(Z.sub(0), Constant(0.0), Ue_boundary),
             DirichletBC(Z.sub(1), Constant(0.0), Ue_boundary)]
    
    #   Obtaining initial guess, zero, by interpolating zero  
    initial_const = Constant((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    u_k = interpolate(initial_const, Z)
    (U1_k, U2_k, K1_k, K2_k, P1_k, P2_k) = split(u_k)      # _k variables only REFER to the associated part of u_k     
    #-------------------------------- dimensions of each part -------------------------------------
    (U1, U2, K1, K2, P1, P2) = u_k.split(deepcopy=True)   # deepcopy is needed, otherwise following np.sizes will returen size of u_k
    n_1 = np.size(U1.vector().get_local()) + np.size(U2.vector().get_local())
    n_c = np.size(K1.vector().get_local()) + np.size(K2.vector().get_local())
    n_d = np.size(P1.vector().get_local()) + np.size(P2.vector().get_local())
    print("n_1 = %s,    n_c = %s,   n_d = %s" % (n_1,n_c,n_d))
    #----------------------------------------------------------------------------------------------
    DoF = n_1 + n_c + n_d

    # Defining bilinear forms
            
    def DPbb(K1, K2, M1, M2, Pi1, Pi2):
        """ the term containing the derivative of the constitutive equation. Ki are solutions
        from the previous step (i.e. the point that the derivative is calculated), Mi are trail
        functions (i.e. unknown of the linearized problem), and Pii are test functions.
        The parameters mu and lam are defined outside, in the main program"""
        F = as_tensor([[1.0 + K1[0], K1[1]], [K2[0], 1.0 + K2[1]]])     # deformation gradient
        FinvT = inv(F.T)
        Mmat = as_tensor([[M1[0], M1[1]], [M2[0], M2[1]]]) 
        PImat = as_tensor([[Pi1[0], Pi1[1]], [Pi2[0], Pi2[1]]])
        return inner(mu*Mmat + (mu - 2*lam*ln(det(F.T*F)))*FinvT*Mmat.T*FinvT + 4*lam*tr(inv(F)*Mmat)*FinvT, PImat)
    
    
    (V1, V2, M1, M2, Q1, Q2) = TrialFunctions(Z)    # the increment at each step
    (Up1, Up2, La1, La2, Pi1, Pi2) = TestFunctions(Z) 
    
    # Material-dependent bilnear form 
    MatDepBF = ( inner(Q1, grad(Up1)) + inner(Q2, grad(Up2)) + inner(grad(V1), La1) + inner(grad(V2), La2)
         - inner(M1, La1) - inner(M2, La2) + DPbb(K1_k, K2_k, M1, M2, Pi1, Pi2)
         - inner(Q1, Pi1) - inner(Q2, Pi2) )*dx
    # Bilnear form for forming the inner product matrix
    InPrBF = ( V1*Up1 + V2*Up2 + inner(grad(V1), grad(Up1)) + inner(grad(V2), grad(Up2)) 
             + inner(M1, La1) + inner(M2, La2) + inner(curl(M1), curl(La1)) + inner(curl(M2), curl(La2))
             + inner(Q1, Pi1) + inner(Q2, Pi2) + inner(div(Q1), div(Pi1)) + inner(div(Q2), div(Pi2)) )*dx
    # This will not be used, just as an input for assemble_system
    RHS = (Constant(0.0)*Up1)*dx
    
    # Assemble matrices of the bilinear forms
    if Guass_points:
        parameters["form_compiler"]["quadrature_degree"] = Guass_points # number of points for Guassian Quadrature
    
    # Note that using assemble() and then bc.apply() will make only rows of boundary dofs to be zero and the 
    # main diagonal element 1. Using assemble_system(), both rows and columns of boundary dofs will set to zero, but
    # the main diagonal may be different than one. We apply bc.apply() one more time to make the main diagonal elements 1 
    
    #Assembled matrices
    S, D = PETScMatrix(), PETScMatrix()
    B = PETScVector()
    assemble_system(MatDepBF, RHS, bcs_h, A_tensor=S, b_tensor=B)
    assemble_system(InPrBF, RHS, bcs_h, A_tensor=D, b_tensor=B)
    
    
    #S = assemble(MatDepBF, form_compiler_parameters={"reorder_dofs_serial" : False})
    #D = assemble(InPrBF)    
   
    for bc in bcs_h:
        bc.apply(S)        
        bc.apply(D)
    #    bc.zero(S)
    #    bc.zero(D)
    
    BCDoFs = 2*len(bcs_h[0].get_boundary_values())
    print("number of boundary dofs is", BCDoFs)
        
    #obtaining FM = D * S.T * D * S 
    from petsc4py.PETSc import Mat
    
    ST = PETScMatrix(S.mat().transpose(Mat()))
    DST = PETScMatrix(D.mat().matMult(ST.mat()))
    #DST = PETScMatrix(D.mat().matMult(S.mat().transpose(Mat())))
    DS = PETScMatrix(D.mat().matMult(S.mat()))
    FM = PETScMatrix(DST.mat().matMult(DS.mat()))
    
           
    # For check: only SVD of S 
    #ST = PETScMatrix(S.mat().transpose(Mat()))
    #FM = PETScMatrix(ST.mat().matMult(S.mat()))
    
    # For check: only SVD of D 
    #FM = PETScMatrix(D.mat().matMult(D.mat()))
    
    #np.set_printoptions(linewidth=300)
    #np.set_printoptions(precision=1)
    #np.set_printoptions(suppress=True)
    #print("\n S = \n", S.array(), "\n \n")
    #print("\n D = \n", D.array(), "\n \n")
    #print("\n Final matrix = \n", FM.array(), "\n \n")
   
    #---------------------- eig with FEniCS SLEPc----------------------------
        
    # Create eigensolver
    eigensolver = SLEPcEigenSolver(FM)
    eigensolver.parameters["solver"] = "krylov-schur"           # Specify the solution method (default is krylov-schur)
    eigensolver.parameters["spectrum"] = "smallest magnitude"   # Specify the part of the spectrum desird
    #eigensolver.parameters["problem_type"] = "gen_hermitian"    # Specify the problem type (this can make a big difference)
    eigensolver.parameters["spectral_transform"] = "shift-and-invert"   # Use the shift-and-invert spectral transformation
    eigensolver.parameters["spectral_shift"] = 1.0e-7          # Specify the shift
    
    #n_eig = n_1                       # the number of eigenvalues to be calculated
    n_eig = 5                       # the number of eigenvalues to be calculated
    eigensolver.solve(n_eig) # Compute the eigenvalues.
    nconv = eigensolver.get_number_converged()                 # Check the number of eigenvalues that converged.
    #print("Number of eigenvalues successfully computed: ", nconv)
         
    eigenvalues = []
    for i in range(nconv):
        r, c = eigensolver.get_eigenvalue(i)    # real and complex part of eigenvalue
        eigenvalues.append(r)
    print("eigenvalues = ", eigenvalues) 
               
    val = np.array(eigenvalues).real.min()
    print("\n Smallest eigen value is ", val)
    BETA_h = pow(val.real, 0.5)
                   
    return BETA_h, mesh.hmax(), DoF
                

# Perform experiments
degreeCG = 1; degreeCurl = 1; degreeDiv = 1      # degrees of CG, curl and div finite elements

# Material properties
mu = 1; lam = 1                                  # Parameters of Neo-Hookean materials

# number of Gaussian points
Guass_points = 0    # to use FEnicCS default value enter 0; usually 10 is fine 

h = []  # element sizes
beta = []  # inf-sup lower bound
DOFs = [] # total number of degrees of freedom 
for n in [1.9, 3.4, 4.9]:
#for n in [2, 4, 6]:
    beta_h, hmesh, DOF_M = compute(n, degreeCG, degreeCurl, degreeDiv)
    h.append(hmesh)
    beta.append(beta_h)
    DOFs.append(DOF_M)

# Convergence rates
from math import log as ln  # log is a dolfin name too

for i in range(0, len(h)):
    Beta_s = ln(beta[i]/beta[i-1])/ln(h[i]/h[i-1])     # beta = O(h^{Beta_s})
       
    print('h=%.3f | DOF = %d | beta_h=%8.4E Rate=%.2f' % (h[i], DOFs[i], beta[i], Beta_s))

TotR = ln(beta[-1]/beta[-3])/ln(h[-1]/h[-3])     # rate using more seperated points 
print('Total rate =%.2f ' % TotR)
