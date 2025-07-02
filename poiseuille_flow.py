import fenics as fe 
import numpy as np
import matplotlib.pyplot as plt

# Define physical parameters
T = 10.0
num_steps = 500
dt = T/num_steps
rho = 1
mu = 1

L_x = 4.0
H = 1.0 # channel height
nx = 10
ny = nx
mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(L_x, H), nx, ny)

V = fe.VectorFunctionSpace(mesh, "P", 2)
Q = fe.FunctionSpace(mesh, "P", 1)

# Define boundaries 
inflow = "near(x[0], 0)" # ie inflow boundary at x[0] = 0(ie x=0)
outflow = "near(x[0], 4.0)" # ie outflow boundary
no_slip = "near(x[1], 0.0) || near(x[1], 1.0)" # For no-slip at walls


# Define boundary conditions 
bcu_noslip = fe.DirichletBC(V, fe.Constant( (0.0, 0.0) ), no_slip)
bcp_inflow = fe.DirichletBC(Q, fe.Constant(8), inflow)
bcp_outflow = fe.DirichletBC(Q, fe.Constant(0), outflow)
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]


# Define trial and test functions for bilinear and linear forms
u_star = fe.TrialFunction(V)
u_np1 = fe.TrialFunction(V)
v = fe.TestFunction(V)

p_np1 = fe.TrialFunction(Q)
q = fe.TestFunction(Q)

# Functions for solutions at previous timestep and just 
# computed solutions
u_n = fe.Function(V) # Velocity from previous solution
u_star_jc = fe.Function(V) # Just computed velocity
u_np1_jc = fe.Function(V)
p_n = fe.Function(Q) # Pressure from previous solution
p_jc = fe.Function(Q) # Just computed pressure 


# Expression used in variational forms
U_cn = 0.5*( u_star + u_n )
n = fe.FacetNormal(mesh)
#mu = fe.Constant(mu)
k = dt #fe.Constant(dt)
#rho = fe.Constant(rho)

# Define strain-rate tensor
def epsilon(u):
    return fe.sym(fe.nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 0.5*mu*epsilon(u) - p*fe.Identity( len(u) )


# Bilinear and linear forms, step 1
a1 = rho*(1/k)*fe.dot(u_star - u_n, v)*fe.dx 
#a1 = rho*(1/k)*fe.dot( (u_star - u_n), v)*fe.dx
    # + fe.inner( sigma(U_cn, p_n), fe.nabla_grad(v) )*fe.dx\
    #         - mu*fe.dot( fe.nabla_grad(U_cn)*n , v)*fe.ds\
                
L1 = rho*fe.dot( fe.dot( u_n, fe.grad(u_n) ), v)*fe.dx + fe.dot(p_n*n, v)*fe.dx

# Bilinear and linear forms, step 2
a2 = fe.dot(fe.nabla_grad(p_np1), fe.nabla_grad(q) )*fe.dx
L2 = fe.dot(fe.nabla_grad(p_n), fe.nabla_grad(q) )*fe.dx\
    - (1/dt)*fe.div(u_star_jc)*q*fe.dx 
    
# Bilinear and linear forms, step 3
a3 = rho*(1/k)*fe.dot( u_np1, v )*fe.dx
L3 = fe.dot(u_star_jc, v)*fe.dx\
    -dt*fe.dot( ( fe.grad(p_jc) - fe.grad(p_n) ), v)*fe.dx
       
    
# Assemble matrices
A1 = fe.assemble(a1)
A2 = fe.assemble(a2)
A3 = fe.assemble(a3)

# # Apply BCs to matrices
# [bc.apply(A1) for bc in bcu]
# [bc.apply(A2) for bc in bcp]

# # Time-stepping
# t = 0
# for n in range(num_steps):
    
#     # Update current time
#     t += dt
    
#     # Step 1: Compute tentative velocity
#     b1 = fe.assemble(L1)
#     [bc.apply(b1) for bc in bcu]
    
#     fe.solve(A1, u_star_jc.vector(), b1)
    
#     # Step 2: Pressure correction
#     b2 = fe.assemble(L2)
#     [bc.apply(b2) for bc in bcp]
    
#     fe.solve(A2, p_jc.vector(), b2)
    
    
#     # Step 3: Compute updated velocity
#     b3 = fe.assemble(L3)
#     fe.solve(A3, u_np1_jc.vector(), b3)
    
#     u_n.assign(u_np1_jc)
#     p_n.assign(p_jc)



