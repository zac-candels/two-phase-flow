# =============================================================
# Diffuse -interface hydrodynamic model for droplet evaporation
# =============================================================
from __future__ import print_function
import random
import numpy as np
import math
import fenics as fe
from array import *
import scipy as sp
import scipy.optimize
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os

fe.parameters["std_out_all_processes"] = False
fe.set_log_level(fe.LogLevel.ERROR)

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "Output/ME_mod/figures")
os.makedirs(outDirName, exist_ok=True)


load = False
init_file = 'initial_condition.xml'

Th00 = 90*fe.pi/180
epsc = 0.1
epsT = 0.05
lamd = 1/6
V = (2.4 + 0.1)*lamd**2

def Radius(Th00, epsc, lamd, V):
    arr = np.linspace(0, 10, 1000)
    FF = np.zeros((arr.size))
    c = 0
    def F(x):
        Th = Th00 + epsc*np.cos(2*np.pi*x)
        return V - (x**2/2)*(2*Th - np.sin(2*Th))/(np.sin(Th))**2
    for idx in arr:
        FF[c] = F(idx)
        c += 1
    F1 = FF[0:-1]*FF[1:]
    filt = F1 < 0
    R00 = arr[1:][filt]
    rd = np.zeros(R00.size)
    for i in range(R00.size):
        rd[i] = sp.optimize.newton(F, R00[i])
    return rd[0]

d = Radius(Th00, epsc, lamd, V)

x0 = 3.0*lamd
Th = Th00 + epsc*np.cos(2*np.pi*d)
R0 = d/np.sin(Th)
Y0 = d/np.tan(Th)

Ch = 1.0e-02
dt = 1.0e-02
mob = 3*Ch**2
theta = 0.5
Rey = 1
Web = 0.2

Cn = fe.Constant(1/Ch)
k = fe.Constant(dt)
We = fe.Constant(1/Web)
Re = fe.Constant(1/Rey)
Pe = fe.Constant(mob)

fe.parameters["form_compiler"]["optimize"] = True
fe.parameters["form_compiler"]["cpp_optimize"] = True

TT = 0.5
RR = 1

domain_n_points = 80
domain_points = []

mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(RR, TT), 80, 80)

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
ME = fe.FunctionSpace(mesh, P1*P1, constrained_domain=pbc)

c_mu_trial = fe.TrialFunction(ME)
q, v = fe.TestFunctions(ME)
vel_trial = fe.TrialFunction(W)
p = fe.TrialFunction(P)
w = fe.TestFunction(W)
r = fe.TestFunction(P)

c_mu_nP1 = fe.Function(ME)
vel_star = fe.Function(W)
p1 = fe.Function(P)
c_mu_n = fe.Function(ME)
vel_n = fe.Function(W)
vel_nP1 = fe.Function(W)

dc, dmu = fe.split(c_mu_trial)
c_nP1, mu_nP1 = fe.split(c_mu_nP1)
c_n, mu_n = fe.split(c_mu_n)

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


u_init = InitialConditions(Cn_val=Cn, R0=R0, x0=x0, Y0=Y0, degree=2)
c_mu_nP1.interpolate(u_init)
c_mu_n.interpolate(u_init)


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

zeta = np.sqrt(2)/3
Wetting = fe.Expression('zeta*cos( Th00 )',
                     zeta=zeta, Th00=Th00, degree=1)

c_var = fe.variable(c_nP1)
f1 = 1/4*(1 - c_var**2)**2
dfdc = fe.diff(f1, c_var)
surf_ten_force = -c_nP1*fe.grad(mu_nP1)

def epsilon(u):
    return 0.5*(fe.nabla_grad(u) + fe.nabla_grad(u).T)

mu_mid = (1-theta)*mu_n + theta*mu_nP1
c_mid = (1-theta)*c_n + theta*c_nP1

def L():
    L0 = fe.inner(c_nP1 - c_n, q)*fe.dx + dt*fe.inner(fe.dot(vel_n, fe.grad(c_mid)), q)*fe.dx + \
         Pe*dt*fe.inner(fe.grad(mu_mid), fe.grad(q))*fe.dx
    LL1 = mu_nP1*v*fe.dx - dfdc*Cn*v*fe.dx - Ch*fe.dot(fe.grad(v), fe.grad(c_nP1))*fe.dx + Wetting*v*ds(1)
    return L0 + LL1

F1 = (1/k)*fe.inner(vel_trial - vel_n, w)*fe.dx + fe.inner(fe.grad(vel_n)*vel_n, w)*fe.dx + \
     Re*fe.inner(2*epsilon(vel_trial), epsilon(w))*fe.dx - We*fe.inner(surf_ten_force, w)*fe.dx

NS_bilin = fe.lhs(F1)
NS_lin = fe.rhs(F1)

pres_update_bilin = fe.inner(fe.grad(p), fe.grad(r))*fe.dx
pres_update_lin = -(1/k)*fe.div(vel_star)*r*fe.dx

vel_update_bilin = fe.inner(vel_trial, w)*fe.dx
vel_update_lin = fe.inner(vel_star, w)*fe.dx - k*fe.inner(fe.grad(p1), w)*fe.dx

class CahnHilliardEquation1(fe.NonlinearProblem):
    def __init__(self, a, L):
        super().__init__()
        self.L = L
        self.a = a
    def F(self, b, x):
        fe.assemble(self.L, tensor=b)
    def J(self, A, x):
        fe.assemble(self.a, tensor=A)

def assemble_CH():
    a_form = fe.derivative(L(), c_mu_nP1, c_mu_trial)   # Jacobian form
    L_form = L()                      # Residual form
    return CahnHilliardEquation1(a_form, L_form)

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
    file = fe.File(file_name + "/final_u_.xml")

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
        c_mu_n.vector()[:] = c_mu_nP1.vector()

        solver.solve(assemble_CH(), c_mu_nP1.vector())

        NS_rhs_vec = fe.assemble(NS_lin)
        for bc in bcu: bc.apply(NS_mat, NS_rhs_vec)
        fe.solve(NS_mat, vel_star.vector(), NS_rhs_vec, "gmres", prec)

        pres_update_rhs_vec = fe.assemble(pres_update_lin)
        fe.solve(pres_update_mat, p1.vector(), pres_update_rhs_vec, "gmres", prec)

        vel_update_rhs_vec = fe.assemble(vel_update_lin)
        for bc in bcu: bc.apply(vel_update_mat, vel_update_rhs_vec)
        fe.solve(vel_update_mat, vel_nP1.vector(), vel_update_rhs_vec, "gmres", prec)

        vel_n.assign(vel_nP1)
        it += 1
        t += dt
        
        if ctr % 100 == 0:
            coords = mesh.coordinates()
            phi_vals = c_mu_nP1.split()[0].compute_vertex_values(mesh)
            triangles = mesh.cells()  # get mesh connectivity
            triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
        
            plt.figure(figsize=(6,5))
            plt.tricontourf(triang, phi_vals, levels=50, cmap="RdBu_r")
            plt.colorbar(label=r"$\phi$")
            plt.title(f"phi at t = {t:.2f}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.tight_layout()
            
            # Save the figure to your output folder
            out_file = os.path.join(outDirName, f"phi_t{ctr:05d}.png")
            plt.savefig(out_file, dpi=200)
            #plt.show()
            plt.close()

        if t >= ts[itc]:
            cfile << (c_mu_nP1.split()[0], t)
            mfile << (c_mu_nP1.split()[1], t)
            yfile << vel_star
            pfile << p1
            itc += 1

    file << c_mu_nP1

def main():
    Tfinal = 1000
    Nsaved = 200
    file_name = "Output/ME_mod"
    droplet_solution(Tfinal, Nsaved, file_name)

if __name__ == "__main__":
    main()
