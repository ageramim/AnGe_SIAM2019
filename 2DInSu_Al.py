""" Arzhang Angoshtari, Ali Gerami, March 2020
This code check the neseccary inf-sup condition for the first mixed formulation 
of nonlinear elasticity by calculating the rank of the associated matrices

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
    errors and convergence rates
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
    
    # trial and test spaces for the first inf-sup condition alpha
    ZTrial_al = FunctionSpace(mesh, MixedElement([DivE, DivE]))
    ZTest_al = FunctionSpace(mesh, MixedElement([CGE, CGE]))

    
    # Just for calculating dimensions n_1, n_c, n_d...
    ZTot = FunctionSpace(mesh, MixedElement([CGE, CGE, CurlE, CurlE, DivE, DivE]))
    #   Obtaining initial guess, zero, by interpolating zero  
    initial_const = Constant((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    u_k = interpolate(initial_const, ZTot)
    (U1_k, U2_k, K1_k, K2_k, P1_k, P2_k) = split(u_k)      # _k variables only REFER to the associated part of u_k     
    #-------------------------------- dimensions of each part -------------------------------------
    (U1, U2, K1, K2, P1, P2) = u_k.split(deepcopy=True)   # deepcopy is needed, otherwise following np.sizes will returen size of u_k
    n_1 = np.size(U1.vector().get_local()) + np.size(U2.vector().get_local())
    n_c = np.size(K1.vector().get_local()) + np.size(K2.vector().get_local())
    n_d = np.size(P1.vector().get_local()) + np.size(P2.vector().get_local())
    print("n_1 = %s,    n_c = %s,   n_d = %s" % (n_1,n_c,n_d))
    #----------------------------------------------------------------------------------------------
        
    ## Define boundary conditions
    #def Ue_boundary(x, on_boundary):
    #    return on_boundary
            
    ##       Homogeneous BCs 
    #bcs_h = [DirichletBC(ZTest_al.sub(0), Constant(0.0), Ue_boundary),
    #         DirichletBC(ZTest_al.sub(1), Constant(0.0), Ue_boundary)]
    
    # Define nonlinear functions             
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
       
    # Definiting bilinear forms
    (Q1_al, Q2_al) = TrialFunctions(ZTrial_al)    
    (Up1_al, Up2_al) = TestFunctions(ZTest_al)        
    BLF_al = ( inner(Q1_al, grad(Up1_al)) + inner(Q2_al, grad(Up2_al)))*dx
     
    # Assemble matrices of the bilinear forms
    if Guass_points:
        parameters["form_compiler"]["quadrature_degree"] = Guass_points # number of points for Guassian Quadrature
           
    BM_al = assemble(BLF_al)
    
    #print("Before applying BCs, the size of the alpha matrix is", BM_al.array().shape)
    #bcs_h[0].apply(BM_al)
    #bcs_h[1].apply(BM_al)
    #for bc in bcs_h:
    #    bc.apply(BM_al)
    
    Mat_al = BM_al.array()
    print("The size of the alpha matrix is", Mat_al.shape)
    Full_rank = Mat_al.shape[0] # rank if surjective
    
    Rank_alp = np.linalg.matrix_rank(Mat_al)
    print("The rank of the alpha matrix is %d out of %d" % (Rank_alp, Full_rank))
    
    Ratio = Rank_alp / Full_rank   
    
    return Ratio, mesh.hmax()
                

# Perform experiments
degreeCG = 2; degreeCurl = 1; degreeDiv = 1      # degrees of CG, curl and div finite elements

# Material properties
mu = 1; lam = 1                                  # Parameters of Neo-Hookean materials

# number of Gaussian points
Guass_points = 0    # to use FEnicCS default value enter 0

h = []  # element sizes
Ratio_al = []  # number of zero SVDs

for n in [1.9, 3.4, 4.9]:
#for n in [2, 4, 6, 8]:
#for n in [2]:
#for n in [4, 8, 12, 16]:
#for n in [3, 7, 11, 15]:
    R_al, hmesh = compute(n, degreeCG, degreeCurl, degreeDiv)
    h.append(hmesh)
    Ratio_al.append(R_al)

    
for i in range(0, len(h)):
    print('h=%.3f | the rank ratio = %.3f' % (h[i], Ratio_al[i]))

