import fenics as fe 
import numpy as np
import matplotlib.pyplot as plt

# NOTE, IF NOT USING VARIATIONAL FORM NOTATION (ie F(.,.) = ),
# THEN NON-TRIAL/TEST FUNCTIONS CANNOT APPEAR IN BILINEAR FORM,
# ONLY IN LINEAR FORMS.

plt.close('all')

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
#mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(L_x, H), nx, ny)
mesh = fe.UnitSquareMesh(16, 16)

V = fe.VectorFunctionSpace(mesh, "P", 2)
Q = fe.FunctionSpace(mesh, "P", 1)

# Define boundaries 
inflow = "near(x[0], 0)" # ie inflow boundary at x[0] = 0(ie x=0)
outflow = "near(x[0], 1.0)" # ie outflow boundary
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
n = fe.FacetNormal(mesh)
mu = fe.Constant(mu)
k = fe.Constant(dt)
rho = fe.Constant(rho)
f = fe.Constant( (0,0) )

# Define strain-rate tensor
def epsilon(u):
    return fe.sym(fe.nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 0.5*mu*epsilon(u) - p*fe.Identity( len(u) )


# Bilinear and linear forms, step 1
a1 = rho*(1/k)*fe.dot( u_star, v )*fe.dx\
     + fe.inner( mu*epsilon(u_star), epsilon(v) )*fe.dx\
             - 0.5*mu*fe.dot( fe.nabla_grad(u_star)*n , v)*fe.ds\
                
L1 = rho*(1/k)*fe.dot(u_n,v)*fe.dx\
    - rho*fe.dot( fe.dot( u_n, fe.nabla_grad(u_n) ), v)*fe.dx\
        - fe.dot(p_n*n, v)*fe.ds\
            - fe.inner( mu*epsilon(u_n), epsilon(v) )*fe.dx\
            + fe.inner( p_n*fe.Identity( len(u_n) ),\
                             epsilon(v) )*fe.dx\
                    + 0.5*mu*fe.dot( fe.nabla_grad(u_n)*n , v)*fe.ds + fe.dot(f, v)*fe.dx
        

# Bilinear and linear forms, step 2
a2 = fe.dot(fe.nabla_grad(p_np1), fe.nabla_grad(q) )*fe.dx
L2 = fe.dot(fe.nabla_grad(p_n), fe.nabla_grad(q) )*fe.dx\
     - (1/k)*fe.div(u_star_jc)*q*fe.dx 
    
# # Bilinear and linear forms, step 3
a3 = fe.dot( u_np1, v )*fe.dx
L3 = fe.dot(u_star_jc, v)*fe.dx\
     -k*fe.dot( ( fe.nabla_grad(p_jc - p_n) ), v)*fe.dx
       
    
# Assemble matrices
A1 = fe.assemble(a1) 
A2 = fe.assemble(a2)
A3 = fe.assemble(a3)

# Apply BCs to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Time-stepping
t = 0
for n in range(num_steps):
    
    # Update current time
    t += dt
    
    # Step 1: Compute tentative velocity
    b1 = fe.assemble(L1)
    [bc.apply(b1) for bc in bcu]
    
    fe.solve(A1, u_star_jc.vector(), b1)
    
    # Step 2: Pressure correction
    b2 = fe.assemble(L2)
    [bc.apply(b2) for bc in bcp]
    
    fe.solve(A2, p_jc.vector(), b2)
    
    
    # Step 3: Compute updated velocity
    b3 = fe.assemble(L3)
    fe.solve(A3, u_np1_jc.vector(), b3)
    
    u_n.assign(u_np1_jc)
    p_n.assign(p_jc)

# Plot velocity field with larger arrows
# Plot velocity field with larger arrows
coords = V.tabulate_dof_coordinates()[::2]  # Shape: (1056, 2)
u_values = u_n.vector().get_local().reshape((V.dim() // 2, 2))  # Shape: (1056, 2)
x = coords[:, 0]  # x-coordinates
y = coords[:, 1]  # y-coordinates
u_x = u_values[:, 0]  # x-components of velocity
u_y = u_values[:, 1]  # y-components of velocity

# Define arrow scale based on maximum velocity
max_u = np.max(np.sqrt(u_x**2 + u_y**2))
arrow_length = 0.05  # 5% of domain size
scale = max_u / arrow_length if max_u > 0 else 1

# Create quiver plot
plt.figure()
M = np.hypot(u_x, u_y)
plt.quiver(x, y, u_x, u_y, M, scale=scale, scale_units='height')
plt.title("Velocity field at final time")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot velocity profile at x=0.5 (unchanged, assuming it works)
num_points = 100
y_values = np.linspace(0, 1, num_points)
x_fixed = 0.5
points = [(x_fixed, y) for y in y_values]
u_x_values = []
for point in points:
    u_at_point = u_n(point)
    u_x_values.append(u_at_point[0])
plt.figure()
plt.plot(u_x_values, y_values)
plt.xlabel("u_x")
plt.ylabel("y")
plt.title("Velocity profile at x=0.5")
plt.show()


