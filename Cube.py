#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Arzhang Angoshtari, Ali Gerami, March 2020
Solving 3D nonlinear elasticity problem using the three-field formulation: 
Find (U,K,P)\in [H^{1}]^{n}*H^{c}*H^{d} such that
            <P, grad(Ups)> =  <B, Ups> + int_{\Gamma_2}\bar{T}. Ups ds  forall Ups \in H^{1}_1
 <grad(U), Lam> - <K, Lam> = 0                                          forall Lam \in H^{c}
   <Pbb(K), Pi > - <P, Pi> = 0                                          forall Pi  \in H^{d}
The output is the convergence rates.
The Newton method (on the PDE level) is employed. The exact solution is
U_e = [0.5*y**3 + 0.5*Coef_e*sin(0.5*pi*y)]. The domain is a unit box. 
BC conditions are imposed as Dirichelt BC on the face Y=0 where U_e is zero 
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
    mesh = UnitCubeMesh(n, n, n)
    
    #       Unstructured mesh
    #Cube = Box(Point(0, 0, 0), Point(1, 1, 1))
    #mesh = generate_mesh(Cube, n)
    
    CGE  = FiniteElement("CG", mesh.ufl_cell(), degreeCG)
    CurlE = FiniteElement("N2curl", mesh.ufl_cell(), degreeCurl)  # N1curl and N2curl
    DivE = FiniteElement("BDM", mesh.ufl_cell(), degreeDiv)        # RT and BDM  
    Z = FunctionSpace(mesh, MixedElement([CGE, CGE, CGE, CurlE, CurlE, CurlE, DivE, DivE, DivE]))
    
    
    # the exact solution  
    Ue_x = Expression('alph * Coef_u0 *(1 + pow(sin(x[0]/R),2)*pow(sin(x[1]/R),2)))', Coef_u0 = Coef_u0, R = R,
                      alph = 1.0, degree = basedegree + degreeCG)    # x-component of displacment
    Ue_y = Expression('alph * Coef_u0 *(1 + pow(sin(x[1]/R),2)*pow(sin(x[2]/R),2)))', Coef_u0 = Coef_u0, R = R,
                      alph = 1.0, degree = basedegree + degreeCG)    # y-component of displacment
    Ue_z = Expression('alph * Coef_u0 *(1 + pow(sin(x[2]/R),2)*pow(sin(x[0]/R),2)))', Coef_u0 = Coef_u0, R = R,
                      alph = 1.0, degree = basedegree + degreeCG)    # z-component of displacment
    Ke_1 = as_vector(((Coef_u0/R) * sin(2*x[0]/R)*pow(sin(x[1]/R),2) , (Coef_u0/R) * sin(2*x[1]/R)*pow(sin(x[0]/R),2) , 0.0)) # the first row of displacement gradient
    
    Ke_2 = as_vector(( 0.0, (Coef_u0/R) * sin(2*x[1]/R)*pow(sin(x[2]/R),2), (Coef_u0/R) * sin(2*x[2]/R)*pow(sin(x[1]/R),2)))  # the second row of displacement gradient
    
    Ke_3 = Ke_1 = as_vector(((Coef_u0/R) * sin(2*x[0]/R)*pow(sin(x[2]/R),2), 0.0, (Coef_u0/R) * sin(2*x[2]/R)*pow(sin(x[0]/R),2))) # 3rd row of displacement gradient

    def_grad = as_tensor([[1.0 + Ke_1[0], Ke_1[1], Ke_1[2]], [Ke_2[0], 1.0 + Ke_2[1], Ke_2[2]], [Ke_3[0], Ke_3[1], 1.0 + Ke_3[2]]])     # deformation gradient
    Stress_ES = mu*def_grad + (2*lam*ln(det(def_grad.T*def_grad)) - mu)*inv(def_grad.T)
    Pe_1 = Stress_ES[0,:]  # 1st row of stress
    Pe_2 = Stress_ES[1,:]  # 2nd row of stress
    Pe_3 = Stress_ES[2,:]  # 3st row of stress
    PP = Stress_ES
    PPP = np.array([[Stress_ES[0,0],Stress_ES[0,1]],[Stress_ES[1,0],Stress_ES[1,1]]]) 
    def Body_f(PP):
        alph =1.0
        return -alph*div(PP)
    
    Be_x = Body_f(PP)[0]   # x-component of the body force for the exact solution
    Be_y = Body_f(PP)[1]   # y-component of the body force for the exact solution
    Be_z = Body_f(PP)[2]`  # z-component of the body force for the exact solution
    
    TBar1 = Stress_ES[0,1,0] + Stress_ES[1,1,0]
    TBar2 = -Stress_ES[0,0,0] - Stress_ES[1,0,0]
    TBar3 = Stress_ES[0,0,0] + Stress_ES[1,0,0]
     
    # Define boundary conditions
    #      Define boundary segments for Dirichlet and traction BCs
    #      Create mesh function over cell facets
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

    #      Mark boundary facets with Y=0 as subdomain 0
    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-12   # tolerance for coordinate comparisons
            return on_boundary and abs(x[1]) < tol
    Gamma_B = BottomBoundary()
    Gamma_B.mark(boundary_parts, 0)

    #       Mark boundary facets with Y=1 as subdomain 1
    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-12   # tolerance for coordinate comparisons
            return on_boundary and abs(x[1] - 1) < tol
    Gamma_T = TopBoundary()
    Gamma_T.mark(boundary_parts, 1)

    #       Mark boundary facet with X=0 as subdomain 2
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-12   # tolerance for coordinate comparisons
            return on_boundary and abs(x[0]) < tol
    Gamma_L = LeftBoundary()
    Gamma_L.mark(boundary_parts, 2)

    #       Mark boundary facet with X=1 as subdomain 3
    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-12   # tolerance for coordinate comparisons
            return on_boundary and abs(x[0] - 1) < tol
    Gamma_R = RightBoundary()
    Gamma_R.mark(boundary_parts, 3)
    
    #       boundary function for the face Y=0
    def bottom_f(x, on_boundary):
        tol = 1E-12   # tolerance for coordinate comparisons
        return on_boundary and abs(x[1]) < tol	
  
    
    #       BCs for the initial guess and the increment in each Newton's iteration

    #           Applying BC with meshfunction as follows does not work for mixed elements apparently
    #           bcs = [DirichletBC(Z.sub(0), Ue_x, boundary_parts, 0),
    #                  DirichletBC(Z.sub(1), Ue_y, boundary_parts, 0)]    
    bcs = [DirichletBC(Z.sub(0), Ue_x, bottom_f),
           DirichletBC(Z.sub(1), Ue_y, bottom_f),
           DirichletBC(Z.sub(2), Ue_z, bottom_f)]
            
    
    # Define nonlinear functions for Newton iteration
    def Pbb(K1, K2, K3, Pi1, Pi2, Pi3):
        """ the term containing the constitutive equation. Ki are trial functions
        and Pii are test functions. The parameters mu and lam are defined outside, 
        in the main program""" 
        F = as_tensor([[1.0 + K1[0], K1[1], K1[2]],
                       [K2[0], 1.0 + K2[1], K2[2]],
                       [K3[0], K3[1], 1.0 + K3[2]]])     # deformation gradient
        PImat = as_tensor([[Pi1[0], Pi1[1], Pi1[2]],
                           [Pi2[0], Pi2[1], Pi2[2]],
                           [Pi3[0], Pi3[1], Pi3[2]]])
        return inner(mu*F + (2*lam*ln(det(F.T*F)) - mu)*inv(F.T), PImat) 
        
    def DPbb(K1, K2, K3, M1, M2, M3, Pi1, Pi2, Pi3):
        """ the term containing the derivative of the constitutive equation. Ki are solutions
        from the previous step (i.e. the point that the derivative is calculated), Mi are trail
        functions (i.e. unknown of the linearized problem), and Pii are test functions.
        The parameters mu and lam are defined outside, in the main program"""
        F = as_tensor([[1.0 + K1[0], K1[1], K1[2]],
                       [K2[0], 1.0 + K2[1], K2[2]],
                       [K3[0], K3[1], 1.0 + K3[2]]])     # deformation gradient
        FinvT = inv(F.T)         
        Mmat = as_tensor([[M1[0], M1[1], M1[2]],
                          [M2[0], M2[1], M2[2]],
                          [M3[0], M3[1], M3[2]]])
        PImat = as_tensor([[Pi1[0], Pi1[1], Pi1[2]],
                           [Pi2[0], Pi2[1], Pi2[2]],
                           [Pi3[0], Pi3[1], Pi3[2]]])
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
    initial_const = Constant((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    u_k = interpolate(initial_const, Z)
    
    (U1_k, U2_k, U3_k, K1_k, K2_k, K3_k, P1_k, P2_k, P3_k) = split(u_k)      # _k variables only REFER to the associated part of u_k
    
    #Test_ini = np.linalg.norm(u_k.vector().get_local(), ord=np.Inf)
    #print("the infinity norm of the initial condition vector is %8.2E" % Test_ini)       

    # Newton iterations
    (V1, V2, V3, M1, M2, M3, Q1, Q2, Q3) = TrialFunctions(Z)    # the increment at each step
    (Up1, Up2, Up3, La1, La2, La3, Pi1, Pi2, Pi3) = TestFunctions(Z)
    
    #           Linearization of the nonlinear problem
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_parts)
    LHS = ( inner(Q1, grad(Up1)) + inner(Q2, grad(Up2)) + inner(Q3, grad(Up3))
            + inner(grad(V1), La1) + inner(grad(V2), La2) + inner(grad(V3), La3)
            - inner(M1, La1) - inner(M2, La2) - inner(M3, La3) + DPbb(K1_k, K2_k, K3_k, M1, M2, M3, Pi1, Pi2, Pi3)
            - inner(Q1, Pi1) - inner(Q2, Pi2) - inner(Q3, Pi3) )*dx
    RHS = - ( inner(P1_k, grad(Up1)) + inner(P2_k, grad(Up2)) + inner(P3_k, grad(Up3))
              + inner(grad(U1_k), La1) + inner(grad(U2_k), La2) + inner(grad(U3_k), La3)
              - inner(K1_k, La1) - inner(K2_k, La2) - inner(K3_k, La3)
              + Pbb(K1_k, K2_k, K3_k, Pi1, Pi2, Pi3) - inner(P1_k, Pi1) - inner(P2_k, Pi2) - inner(P3_k, Pi3)
              - Be_x*Up1 - Be_y*Up2 - Be_z*Up3 )*dx + TBar1*Up1*ds(1) - TBar2*Up2*ds(2) + TBar3*Up2*ds(3)
    
    
    z = Function(Z)     # increment vector
    u = Function(Z)     # the current solution: u = u_k + omega*z
    omega = 0.8         # relaxation parameter, omega<1 may improve convergence in some problems
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
       # Be_x.alph = alpha1; TBar.alph = alpha1     # increasing body force gradually 
                
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
    
    # Calculating errors
    (U1, U2, U3, K1, K2, K3, P1, P2, P3) = u_k.split()
    #Er_U = np.sqrt(errornorm(Ue_x, U1, norm_type="L2")**2
    #               + errornorm(Ue_y, U2, norm_type="L2")**2)  # L2 norm of displacement 
    #Er_K = np.sqrt(errornorm(Ke_1, K1, norm_type="L2")**2
    #               + errornorm(Ke_2, K2, norm_type="L2")**2)  # L2 norm of displacement gradient
    #Er_P = np.sqrt(errornorm(Pe_1, P1, norm_type="L2")**2
    #               + errornorm(Pe_2, P2, norm_type="L2")**2)  # L2 norm of stress
    Er_U = np.sqrt(errornorm(Ue_x, U1, norm_type="L2", degree_rise= degRise)**2
                   + errornorm(Ue_y, U2, norm_type="L2", degree_rise= degRise)**2
                   + errornorm(Ue_z, U3, norm_type="L2", degree_rise= degRise)**2)  # L2 norm of displacement 
    Er_K = np.sqrt(errornorm(Ke_1, K1, norm_type="L2", degree_rise= degRise)**2
                   + errornorm(Ke_2, K2, norm_type="L2", degree_rise= degRise)**2
                   + errornorm(Ke_3, K3, norm_type="L2", degree_rise= degRise)**2)  # L2 norm of displacement gradient
    Er_P = np.sqrt(errornorm(Pe_1, P1, norm_type="L2", degree_rise= degRise)**2
                   + errornorm(Pe_2, P2, norm_type="L2", degree_rise= degRise)**2
                   + errornorm(Pe_3, P3, norm_type="L2", degree_rise= degRise)**2)  # L2 norm of stress
    
    Er = [Er_U, Er_K, Er_P]
    
    return Er, mesh.hmax()
                

# Perform experiments
degreeCG = 1; degreeCurl = 1; degreeDiv = 1      # degrees of CG, curl and div finite elements

# Material properties
mu = 20; lam = 400000                                  # Parameters of Neo-Hookean materials

# exact solution
Coef_e = 1.0        # set to 0.0 or 1.0. If 0.0, then exact solution will be a polynomial

# degree of interpolation for expressions is the basedegree plus the degree of the associated FE
basedegree = 2

# degree_rise for calculation of errornorms
degRise = 4

# number of Gaussian points
Guass_points = 10    # to use FEnicCS default value enter 0


h = []  # element sizes
E = []  # errors
for n in [3,4]:
    ErrM, hmesh = compute(n, degreeCG, degreeCurl, degreeDiv)
    h.append(hmesh)
    E.append(ErrM) 

# Convergence rates
from math import log as ln  # log is a dolfin name too

for i in range(1, len(E)):
    UR = ln(E[i][0]/E[i-1][0])/ln(h[i]/h[i-1])     #the convergenece rate of displacement
    KR = ln(E[i][1]/E[i-1][1])/ln(h[i]/h[i-1])     #the convergenece rate of displecement gradient
    PR = ln(E[i][2]/E[i-1][2])/ln(h[i]/h[i-1])     #the convergenece rate of stress
    
    print('h=%.3f | U_Error=%8.4E U_ConvRate=%.2f | K_Error=%8.4E K_ConvRate=%.2f | P_Error=%8.4E P_ConvRate=%.2f'
          % (h[i], E[i][0], UR, E[i][1], KR, E[i][2], PR))
    

