#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Arzhang Angoshtari, Ali Gerami, Match 2020
Solving 2D nonlinear elasticity problem using the three-field formulation: 
Find (U,K,P)\in [H^{1}]^{n}*H^{c}*H^{d} such that
            <P, grad(Ups)> =  <B, Ups> + int_{\Gamma_2}\bar{T}. Ups ds  forall Ups \in H^{1}_1
 <grad(U), Lam> - <K, Lam> = 0                                          forall Lam \in H^{c}
   <Pbb(K), Pi > - <P, Pi> = 0                                          forall Pi  \in H^{d}
The output is the convergence rates.
The Newton method (on the PDE level) is employed. The exact solution is
U_e = [0.5*y**3 + 0.5*Coef_e*sin(0.5*pi*y)]. The domain is a unit box. 
BC conditions are imposed as Dirichelt BC on the bottom face where U_e is zero 
and the traction BC are imposed on the remaining faces. 
Traction BC is imposed gradually to improve the convergance of Newton's method with steps that are adaptively selected by the program  

Input parameters:
    degreeCG, degreeCurl, degreeDiv: degrees of CG, curl and div finite elements
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
    min_stepsize: minimum allowed step-size for increasing the load factor Newton's method 
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
    mesh = UnitSquareMesh(n, n)
    #       Unstructured mesh
    #Rec = Rectangle(Point(0, 0), Point(1, 1))
    #mesh = generate_mesh(Rec, n-1)
    
    CGE  = FiniteElement("CG", mesh.ufl_cell(), degreeCG)
    CurlE = FiniteElement("N2curl", mesh.ufl_cell(), degreeCurl)  # N1curl and N2curl
    DivE = FiniteElement("BDM", mesh.ufl_cell(), degreeDiv)        # RT and BDM  
    Z = FunctionSpace(mesh, MixedElement([CGE, CGE, CurlE, CurlE, DivE, DivE]))
    
    
    # the exact solution  
    Ue_x = Expression('alph * ( 0.5*pow(x[1], 3) + 0.5*Coef_e*sin(0.5*pi*x[1]) )', Coef_e = Coef_e,
                      alph = 1.0, degree = basedegree + degreeCG)    # x-component of displacment
    Ue_y = Constant(0.0)                                 # y-component of displacment
    
    Ke_1 = Expression(("0.0","alph * ( 1.5*x[1]*x[1] + 0.25*Coef_e*pi*cos(0.5*pi*x[1]) )"), Coef_e = Coef_e,
                      alph = 1.0, degree = basedegree + degreeCurl)  # the first row of displacement gradient
    Ke_2 = Constant((0.0,0.0))                           # the second row of displacement gradient
    
    Pe_1 = Expression(("0.0", "alph * ( 1.5*mu*x[1]*x[1] + 0.25*Coef_e*mu*pi*cos(0.5*pi*x[1]) )"), Coef_e = Coef_e,
                      alph = 1.0, mu = mu, degree = basedegree + degreeDiv)  # 1st row of stress
    Pe_2 = Expression(("alph * ( 1.5*mu*x[1]*x[1] + 0.25*Coef_e*mu*pi*cos(0.5*pi*x[1]) )", "0.0"), Coef_e = Coef_e,
                      alph = 1.0, mu = mu, degree = basedegree + degreeDiv)  # 2nd row of stress
    Be_x = Expression('alph * ( (1/8.0)*Coef_e*mu*pi*pi*sin(0.5*pi*x[1]) - 3.0*mu*x[1] )', Coef_e = Coef_e, mu = mu, 
                      alph = 0.0, degree = basedegree + degreeDiv)   # x-component of the body force for the exact solution
    Be_y = Constant(0.0)                                                  # y-component of the body force for the exact solution
    TBar = Expression('alph * ( 1.5*mu*x[1]*x[1] + 0.25*Coef_e*mu*pi*cos(0.5*pi*x[1]) )', Coef_e = Coef_e, mu = mu,
                      alph = 0.0, degree = basedegree + degreeDiv)   # an expression for defining traction boundary conditions
        
    # Define boundary conditions
    #      Define boundary segments for Dirichlet and traction BCs
    #      Create mesh function over cell facets
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

    #      Mark bottom boundary facets as subdomain 0
    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-12   # tolerance for coordinate comparisons
            return on_boundary and abs(x[1]) < tol
    Gamma_B = BottomBoundary()
    Gamma_B.mark(boundary_parts, 0)

    #       Mark top boundary facets as subdomain 1
    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-12   # tolerance for coordinate comparisons
            return on_boundary and abs(x[1] - 1) < tol
    Gamma_T = TopBoundary()
    Gamma_T.mark(boundary_parts, 1)

    #       Mark left boundary as subdomain 2
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-12   # tolerance for coordinate comparisons
            return on_boundary and abs(x[0]) < tol
    Gamma_L = LeftBoundary()
    Gamma_L.mark(boundary_parts, 2)

    #       Mark right boundary as subdomain 3
    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-12   # tolerance for coordinate comparisons
            return on_boundary and abs(x[0] - 1) < tol
    Gamma_R = RightBoundary()
    Gamma_R.mark(boundary_parts, 3)
    
    #       Bottom boundary function
    def bottom_f(x, on_boundary):
        tol = 1E-12   # tolerance for coordinate comparisons
        return on_boundary and abs(x[1]) < tol	
  
    
    #       BCs for the initial guess and the increment in each Newton's iteration

    #           Applying BC with meshfunction as follows does not work for mixed elements apparently
    #           bcs = [DirichletBC(Z.sub(0), Ue_x, boundary_parts, 0),
    #                  DirichletBC(Z.sub(1), Ue_y, boundary_parts, 0)]    
    bcs = [DirichletBC(Z.sub(0), Ue_x, bottom_f),
           DirichletBC(Z.sub(1), Ue_y, bottom_f)]
            
    # Define nonlinear functions for Newton iteration
    def Pbb(K1, K2, Pi1, Pi2):
        """ the term containing the constitutive equation. Ki are trial functions
        and Pii are test functions. The parameters mu and lam are defined outside, 
        in the main program""" 
        F = as_tensor([[1.0 + K1[0], K1[1]], [K2[0], 1.0 + K2[1]]])     # deformation gradient
        PImat = as_tensor([[Pi1[0], Pi1[1]], [Pi2[0], Pi2[1]]])
        return inner(mu*F + (2*lam*ln(det(F.T*F)) - mu)*inv(F.T), PImat) 
        
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
    
    
    ## -------------------- Obtaining the initial guess by solving a linear problem --------------
    ##(U1, U2, K1, K2, P1, P2) = TrialFunctions(Z)
    ##(Up1, Up2, La1, La2, Pi1, Pi2) = TestFunctions(Z)
    
    ##Body = Constant(0.0)
    ##LHS = ( inner(U1, Up1) + inner(U2, Up2) + inner(K1, La1) + inner(K2, La2)
    ##        + inner(P1, Pi1) + inner(P2, Pi2) )*dx
    ##RHS = (Body*Up1 + Body*Up2)*dx       
    
    #            Compute solution
    ##u_k = Function(Z)                                       # initial guess and solution from the previous step
    
    ##if Guass_points:
    ##    parameters["form_compiler"]["quadrature_degree"] = Guass_points  # number of points for Guassian Quadrature    
    ###solve(LHS == RHS, u_k, bcs)
    ##A = assemble(LHS)
    ##b = assemble(RHS)
    ##for condition in bcs: condition.apply(A, b)
    ##solve(A, u_k.vector(), b, 'lu')
    ## -------------------------------------------------------------------------------------------
    
    # Obtaining initial guess, zero, by interpolating zero  
    initial_const = Constant((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    u_k = interpolate(initial_const, Z)
    
    (U1_k, U2_k, K1_k, K2_k, P1_k, P2_k) = split(u_k)      # _k variables only REFER to the associated part of u_k
    
    #Test_ini = np.linalg.norm(u_k.vector().get_local(), ord=np.Inf)
    #print("the infinity norm of the initial condition vector is %8.2E" % Test_ini)       

    # Newton iterations
    (V1, V2, M1, M2, Q1, Q2) = TrialFunctions(Z)    # the increment at each step
    (Up1, Up2, La1, La2, Pi1, Pi2) = TestFunctions(Z)
    
    #           Linearization of the nonlinear problem
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_parts)
    LHS = ( inner(Q1, grad(Up1)) + inner(Q2, grad(Up2)) + inner(grad(V1), La1) + inner(grad(V2), La2)
         - inner(M1, La1) - inner(M2, La2) + DPbb(K1_k, K2_k, M1, M2, Pi1, Pi2)
         - inner(Q1, Pi1) - inner(Q2, Pi2) )*dx
    RHS = - ( inner(P1_k, grad(Up1)) + inner(P2_k, grad(Up2)) + inner(grad(U1_k), La1) + inner(grad(U2_k), La2) 
            - inner(K1_k, La1) - inner(K2_k, La2) + Pbb(K1_k, K2_k, Pi1, Pi2) - inner(P1_k, Pi1) - inner(P2_k, Pi2)
            - Be_x*Up1 - Be_y*Up2 )*dx + TBar*Up1*ds(1) - TBar*Up2*ds(2) + TBar*Up2*ds(3) 


    
    z = Function(Z)     # increment vector
    u = Function(Z)     # the current solution: u = u_k + omega*z
    omega = 1.0         # relaxation parameter, omega<1 may improve convergence in some problems
    tol = 1.0E-5        # convergence tolerance for the infinity norm of the increment z
    maxiter = 15        # maximum allowed iteration for Newton's method
    min_stepsize = 1.0E-4 # minimum allowed step-size 
    
    alpha0 = 0.0        # the parameter that gradually increases the load to get a better convergence behaviour
    u_alpha0 = Function(Z)    # the solution for alpha0. This is required as the solution for new alpha1 may be discarded
    u_alpha0.assign(u_k)      # u_alpha0 initalized as the solution for alpha0 = 0.0
    step_size = 1.0     # alpha0 is increased by step_size at each iteration
    
    while alpha0 < 1.0:
        alpha1 = alpha0 + step_size         #alpha1 will be used to calculate the next step of loading
        if alpha1 > 1.0:
            alpha1 = 1.0; step_size = 1.0 - alpha0      # 1.0 is the largest value for alpha
        
        print("------------- load factor (alpha) = %6.2E------------------" % alpha1)
        Be_x.alph = alpha1; TBar.alph = alpha1     # increasing body force gradually 
                
        eps = 1.0           # An initial value for the infinity norm of u-u_k
        iter = 0            # iteration counter
        while eps > tol and iter < maxiter:
            iter += 1
            if Guass_points:
                parameters["form_compiler"]["quadrature_degree"] = Guass_points # number of points for Guassian Quadrature
            solve(LHS == RHS, z, bcs)
            #Alternative way for solving in theb presence of multiple BCs
            #A = assemble(LHS)
            #b = assemble(RHS)
            #for condition in bcs: condition.apply(A, b)
            #solve(A, z.vector(), b, 'lu')
                        
            eps = np.linalg.norm(z.vector().get_local(), ord=np.Inf)
            print("iter = %d: the infinity norm of the increment is %8.2E" % (iter, eps))
            u.vector()[:] = u_k.vector() + omega*z.vector()
            u_k.assign(u)   # updating for the next iteration
        
        if iter == maxiter:
            print("Newton iterations reached the max iteration, decreasing step-size (n = %d)" % n)
            step_size = 0.5 * step_size     # decreasing step_size
            u_k.assign(u_alpha0)            # rejecting the result of this step and starting again
        else:
            u_alpha0.assign(u_k)            # accepting this step
            alpha0 = alpha1
            step_size = 2.0 * step_size     # increasing the step-size
        
        if step_size < min_stepsize:
            print("Too small step_size! step_size = %6.2E" % step_size)
            import sys
            sys.exit("Termination!")
        
    DOF = np.size(u_k.vector().get_local()) # total number of degrees of freedom
    # Calculating errors
    (U1, U2, K1, K2, P1, P2) = u_k.split()
#    Er_U = np.sqrt(errornorm(Ue_x, U1, norm_type="L2")**2
#                   + errornorm(Ue_y, U2, norm_type="L2")**2)  # L2 norm of displacement 
#    Er_K = np.sqrt(errornorm(Ke_1, K1, norm_type="L2")**2
#                   + errornorm(Ke_2, K2, norm_type="L2")**2)  # L2 norm of displacement gradient
#    Er_P = np.sqrt(errornorm(Pe_1, P1, norm_type="L2")**2
#                   + errornorm(Pe_2, P2, norm_type="L2")**2)  # L2 norm of stress

    Er_U = np.sqrt((errornorm(Ue_x, U1, norm_type="L2", degree_rise= degRise)/norm(Ue_x, mesh = mesh))**2
                   + (errornorm(Ue_y, U2, norm_type="L2", degree_rise= degRise))**2)  # L2 norm of displacement 
    Er_K = np.sqrt((errornorm(Ke_1, K1, norm_type="L2", degree_rise= degRise)/norm(Ke_1, mesh = mesh))**2
                   + (errornorm(Ke_2, K2, norm_type="L2", degree_rise= degRise))**2)  # L2 norm of displacement gradient
    Er_P = np.sqrt((errornorm(Pe_1, P1, norm_type="L2", degree_rise= degRise)/norm(Pe_1, mesh = mesh))**2
                   + (errornorm(Pe_2, P2, norm_type="L2", degree_rise= degRise)/norm(Pe_2, mesh = mesh))**2)  # L2 norm of stress
    
    Er = [Er_U, Er_K, Er_P]
    
    return Er, mesh.hmax(), DOF
                

# Perform experiments
degreeCG = 1; degreeCurl = 1; degreeDiv = 1      # degrees of CG, curl and div finite elements

# Material properties
mu = 80; lam = 400000                                  # Parameters of Neo-Hookean materials

# exact solution
Coef_e = 1.0        # set to 0.0 or 1.0. If 0.0, then exact solution will be a polynomial

# degree of interpolation for expressions is the basedegree plus the degree of the associated FE
basedegree = 2

# degree_rise for calculation of errornorms
degRise = 4

# number of Gaussian points
Guass_points = 0    # to use FEnicCS default value enter 0


h = []  # element sizes
E = []  # errors
DOFs = [] # total number of degrees of freedom 
for n in [8, 10, 12]:
    ErrM, hmesh, DOF_M = compute(n, degreeCG, degreeCurl, degreeDiv)
    h.append(hmesh)
    E.append(ErrM)
    DOFs.append(DOF_M)
    
# Convergence rates
from math import log as ln  # log is a dolfin name too

#for i in range(1, len(E)):
#    UR = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])     #the convergenece rate of displacement
#    KR = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])     #the convergenece rate of displecement gradient
#    PR = ln(E[i][2]/E[i-1][2])/ln(h[i]/h[i-1])     #the convergenece rate of stress
    
#    print('h=%.3f | U_Error=%8.4E U_ConvRate=%.2f | K_Error=%8.4E K_ConvRate=%.2f | P_Error=%8.4E P_ConvRate=%.2f'
#          % (h[i], E[i][0], UR, E[i][1], KR, E[i][2], PR))
    
for i in range(0, len(E)):
    UR = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])     #the convergenece rate of displacement
    KR = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])     #the convergenece rate of displecement gradient
    PR = ln(E[i][2]/E[i-1][2])/ln(h[i]/h[i-1])     #the convergenece rate of stress
    
    print('h=%.3f | DOF = %d | U_Error=%8.4E U_ConvRate=%.2f | K_Error=%8.4E K_ConvRate=%.2f | P_Error=%8.4E P_ConvRate=%.2f'
          % (h[i], DOFs[i], E[i][0], UR, E[i][1], KR, E[i][2], PR))

UR = ln(E[-1][0]/E[-3][0])/ln(h[-1]/h[-3])     #the convergenece rate of displacement
KR = ln(E[-1][1]/E[-3][1])/ln(h[-1]/h[-3])     #the convergenece rate of displecement gradient
PR = ln(E[-1][2]/E[-3][2])/ln(h[-1]/h[-3])     #the convergenece rate of stress
print('Total rates: U_ConvRate=%.2f | K_ConvRate=%.2f | P_ConvRate=%.2f' % (UR, KR, PR))
