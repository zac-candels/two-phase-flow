from __future__ import print_function
import random
import numpy as np
import math
import fenics as fe
from array import *
import scipy as sp
import scipy.optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os

fe.parameters["std_out_all_processes"] = False
fe.set_log_level(fe.LogLevel.ERROR)

theta_deg = 30
theta = theta_deg*np.pi/180
sizeParam = 1.57
epsc = 0.1
epsT = 0.05
lamd = 1/6
V = (2.4 + 0.1)*lamd**2

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"proper_nonDimensionalization_CA{theta_deg}")
matPlotFigs = outDirName + "/matPlotFigs"
os.makedirs(matPlotFigs, exist_ok=True)
os.makedirs(outDirName, exist_ok=True)


fe.parameters["form_compiler"]["optimize"] = True
fe.parameters["form_compiler"]["cpp_optimize"] = True

TT = 0.5
RR = 1

nx, ny = 200, 200
h = np.min(RR/nx, TT/ny)
domain_points = []

mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(RR, TT),
                        nx, ny, diagonal="crossed")


dt1 = 0.1*h**2
dt2 = 0.00001
dt = np.min(dt1, dt2)

Cn = fe.Constant(0.02)
k = fe.Constant(dt)
We = fe.Constant(0.02)
Re = fe.Constant(1)
Pe = fe.Constant(1/(3*Cn**2))

mesh_file = fe.File("mesh.xml")
mesh_file << mesh

class PeriodicBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < fe.DOLFIN_EPS and x[0] > -fe.DOLFIN_EPS and on_boundary)
    def map(self, x, y):
        y[0] = x[0] - RR
        y[1] = x[1]

pbc = PeriodicBoundary()
P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = fe.VectorFunctionSpace(mesh, "Lagrange", 2, constrained_domain=pbc)
P = fe.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)
ME = fe.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)

c_trial = fe.TrialFunction(ME)
mu_trial = fe.TrialFunction(ME)
q = fe.TestFunction(ME)
v = fe.TestFunction(ME)
vel_trial = fe.TrialFunction(W)
p = fe.TrialFunction(P)
w = fe.TestFunction(W)
r = fe.TestFunction(P)

vel_star = fe.Function(W)
p1 = fe.Function(P)
vel_n = fe.Function(W)
vel_nP1 = fe.Function(W)

c_nP1 = fe.Function(ME)
mu_nP1 = fe.Function(ME)
c_n = fe.Function(ME)
mu_n = fe.Function(ME)

class InitialConditions(fe.UserExpression):
    def __init__(self, Cn_val, R0, x0, Y0, **kwargs):
        self.Cn_val = float(Cn_val)   # extract scalar
        self.R0 = R0
        self.x0 = x0
        self.Y0 = Y0
        random.seed(2 + fe.MPI.rank(fe.MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        r = np.sqrt((x[0] - self.x0)**2 + (x[1] )**2)
        dist = r - self.R0
        values[0] = -np.tanh(dist / (np.sqrt(2) * 0.01))
        values[1] = 0.0

    def value_shape(self):
        return (2,)
    
    
c_init_expr = fe.Expression(
    "0.5 - 0.5 * tanh( 2.0 * (sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R) / eps )",
    degree=2,  # polynomial degree used for interpolation
    xc=x0,
    yc=Y0,
    R=R0,
    eps=Cn
)

c_n = fe.interpolate(c_init_expr, ME)



class LowerBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] >= -abs(epsT*lamd) and x[1] <= abs(epsT*lamd)

class TopBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], TT)

class LeftBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[0], 0.0)

class RightBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[0], RR)

mesh_function = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)

Gamma_1 = LowerBoundary()
Gamma_1.mark(mesh_function, 1)
Gamma_2 = TopBoundary()
Gamma_2.mark(mesh_function, 2)
Gamma_3 = LeftBoundary()
Gamma_3.mark(mesh_function, 3)
Gamma_4 = RightBoundary()
Gamma_4.mark(mesh_function, 4)

noslip1 = fe.DirichletBC(W, (0, 0), mesh_function, 1)
noslip2 = fe.DirichletBC(W, (0, 0), mesh_function, 2)
bcu = [noslip1, noslip2]

ds = fe.Measure('ds', domain=mesh, subdomain_data=mesh_function)
n = fe.FacetNormal(mesh)

# Create MeshFunction for boundary markers
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

# Subdomain for bottom wall
class Bottom(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], 0.0)

bottom = Bottom()
bottom.mark(boundaries, 1)   # assign ID = 1 to bottom boundary
ds_bottom = fe.Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=1)


zeta = np.sqrt(2)/3
Wetting = fe.Expression('zeta*cos( theta )',
                     zeta=zeta, theta=theta, degree=1)


surf_ten_force = -c_n*fe.grad(mu_n)

def epsilon(u):
    return 0.5*(fe.nabla_grad(u) + fe.nabla_grad(u).T)

def mobility(phi_n):
    grad_phi_n = fe.grad(phi_n)
    
    abs_grad_phi_n = fe.sqrt(fe.dot(grad_phi_n, grad_phi_n) + 1e-6)
    inv_abs_grad_phi_n = 1.0 / abs_grad_phi_n
    
    mob = (1/Pe)*( 1 - Cn*4*phi_n*(1 - phi_n) * inv_abs_grad_phi_n )
    return mob
    


bilin_form_AC = c_trial * q * fe.dx
bilin_form_mu = mu_trial * v * fe.dx

lin_form_AC = c_n * q * fe.dx - dt*v*fe.dot(vel_n, fe.grad(c_n))*fe.dx\
    - dt*fe.dot(fe.grad(q), mobility(c_n)*fe.grad(c_n))*fe.dx\
        - 0.5*dt**2 * fe.dot(vel_n, fe.grad(q)) * fe.dot(vel_n, fe.grad(c_n)) *fe.dx\
                + dt*Cn*np.cos(theta)*v*mobility(c_n)*4*c_n*(1 - c_n)*ds_bottom

lin_form_mu =  (1/Cn)*( 48*(c_n - 1)*(c_n - 0)*(c_n - 0.5)*v*fe.dx\
    + (3/2)*Cn**2*fe.dot(fe.grad(c_n),fe.grad(v))*fe.dx )


F1 = (1/k)*fe.inner(vel_trial - vel_n, w)*fe.dx + fe.inner(fe.grad(vel_n)*vel_n, w)*fe.dx + \
     (1/Re)*fe.inner(2*epsilon(vel_trial), epsilon(w))*fe.dx - (1/We)*fe.inner(surf_ten_force, w)*fe.dx

NS_bilin = fe.lhs(F1)
NS_lin = fe.rhs(F1)

pres_update_bilin = fe.inner(fe.grad(p), fe.grad(r))*fe.dx
pres_update_lin = -(1/k)*fe.div(vel_star)*r*fe.dx

vel_update_bilin = fe.inner(vel_trial, w)*fe.dx
vel_update_lin = fe.inner(vel_star, w)*fe.dx - k*fe.inner(fe.grad(p1), w)*fe.dx

c_mat = fe.assemble(bilin_form_AC)
mu_mat = fe.assemble(bilin_form_mu)

solver = fe.NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

prec = "amg" if fe.has_krylov_solver_preconditioner("amg") else "default"

def droplet_solution(Tfinal, Nt, file_name):
    mfile = fe.File(file_name + "/Potential/result.pvd", "compressed")
    cfile = fe.File(file_name + "/Phasefield/result.pvd", "compressed")
    yfile = fe.File(file_name + "/Velocity/result.pvd", "compressed")
    pfile = fe.File(file_name + "/Pressure/result.pvd", "compressed")

    NS_mat = fe.assemble(NS_bilin)
    pres_update_mat = fe.assemble(pres_update_bilin)
    vel_update_mat = fe.assemble(vel_update_bilin)

    it = 0
    t = 0.0
    ts = np.linspace(0, Tfinal, Nt)
    itc = 0

    ctr = -1
    while t < Tfinal:
        ctr +=1
        
        rhs_AC = fe.assemble(lin_form_AC)
        rhs_mu = fe.assemble(lin_form_mu)
        
        fe.solve(c_mat, c_nP1.vector(), rhs_AC)
        fe.solve(mu_mat, mu_n.vector(), rhs_mu)

        NS_rhs_vec = fe.assemble(NS_lin)
        for bc in bcu: bc.apply(NS_mat, NS_rhs_vec)
        fe.solve(NS_mat, vel_star.vector(), NS_rhs_vec, "gmres", prec)

        pres_update_rhs_vec = fe.assemble(pres_update_lin)
        fe.solve(pres_update_mat, p1.vector(), pres_update_rhs_vec, "gmres", prec)

        vel_update_rhs_vec = fe.assemble(vel_update_lin)
        for bc in bcu: bc.apply(vel_update_mat, vel_update_rhs_vec)
        fe.solve(vel_update_mat, vel_nP1.vector(), vel_update_rhs_vec, "gmres", prec)

        c_n.assign(c_nP1)
        mu_n.assign(mu_nP1)
        vel_n.assign(vel_nP1)
        it += 1
        t += dt
        
        if ctr % 100 == 0:
            coords = mesh.coordinates()
            phi_vals = c_n.compute_vertex_values(mesh)
            triangles = mesh.cells()  # get mesh connectivity
            triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
        
            plt.figure(figsize=(6,5))
            plt.tricontourf(triang, phi_vals, levels=90, cmap="RdBu_r")
            plt.colorbar(label=r"$\phi$")
            plt.title(f"phi at t = {t:.2f}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.tight_layout()
            
            # Save the figure to your output folder
            out_file = os.path.join(matPlotFigs, f"phi_t{ctr:05d}.png")
            plt.savefig(out_file, dpi=200)
            #plt.show()
            plt.close()

        if t >= ts[itc]:
            cfile << (c_n, t)
            mfile << (mu_n, t)
            yfile << vel_star
            pfile << p1
            itc += 1


def main():
    Tfinal = 1000
    Nsaved = 200
    droplet_solution(Tfinal, Nsaved, outDirName)

if __name__ == "__main__":
    main()

