# =============================================================
# Diffuse -interface hydrodynamic model for droplet evaporation
# =============================================================
from __future__ import print_function
import random
import numpy as np
import math
from dolfin import *
from array import *
import scipy as sp
import scipy.optimize
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os

parameters["std_out_all_processes"] = False
set_log_level(LogLevel.ERROR)

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "Output/ME_mod/figures")
os.makedirs(outDirName, exist_ok=True)


load = False
init_file = 'initial_condition.xml'

Th00 = 90*pi/180
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
    for y in arr:
        FF[c] = F(y)
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

Cn = Constant(1/Ch)
k = Constant(dt)
We = Constant(1/Web)
Re = Constant(1/Rey)
Pe = Constant(mob)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

TT = 0.5
RR = 1

domain_n_points = 80
domain_points = []

mesh = RectangleMesh(Point(0, 0), Point(RR, TT), 80, 80)

mesh_file = File("mesh.xml")
mesh_file << mesh

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)
    def map(self, x, y):
        y[0] = x[0] - RR
        y[1] = x[1]

pbc = PeriodicBoundary()
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = VectorFunctionSpace(mesh, "Lagrange", 2, constrained_domain=pbc)
P = FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)
ME = FunctionSpace(mesh, P1*P1, constrained_domain=pbc)

du = TrialFunction(ME)
q, v = TestFunctions(ME)
y = TrialFunction(W)
p = TrialFunction(P)
w = TestFunction(W)
r = TestFunction(P)

u = Function(ME)
u1 = Function(W)
p1 = Function(P)
u0 = Function(ME)
y0 = Function(W)

dc, dmu = split(du)
c, mu = split(u)
c0, mu0 = split(u0)

class InitialConditions(UserExpression):
    def __init__(self, Cn_val, R0, x0, Y0, **kwargs):
        self.Cn_val = float(Cn_val)   # extract scalar
        self.R0 = R0
        self.x0 = x0
        self.Y0 = Y0
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        r = np.sqrt((x[0] - self.x0)**2 + (x[1] )**2)
        dist = r - self.R0
        values[0] = -np.tanh(dist / (np.sqrt(2) * 0.01))
        values[1] = 0.0

    def value_shape(self):
        return (2,)


u_init = InitialConditions(Cn_val=Cn, R0=R0, x0=x0, Y0=Y0, degree=2)
u.interpolate(u_init)
u0.interpolate(u_init)


class LowerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] >= -abs(epsT*lamd) and x[1] <= abs(epsT*lamd)

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], TT)

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], RR)

mesh_function = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

Gamma_1 = LowerBoundary()
Gamma_1.mark(mesh_function, 1)
Gamma_2 = TopBoundary()
Gamma_2.mark(mesh_function, 2)
Gamma_3 = LeftBoundary()
Gamma_3.mark(mesh_function, 3)
Gamma_4 = RightBoundary()
Gamma_4.mark(mesh_function, 4)

noslip1 = DirichletBC(W, (0, 0), mesh_function, 1)
noslip2 = DirichletBC(W, (0, 0), mesh_function, 2)
bcu = [noslip1, noslip2]

ds = Measure('ds', domain=mesh, subdomain_data=mesh_function)
n = FacetNormal(mesh)

zeta = np.sqrt(2)/3
Wetting = Expression('zeta*cos( Th00 )',
                     zeta=zeta, Th00=Th00, degree=1)

c_var = variable(c)
f1 = 1/4*(1 - c_var**2)**2
dfdc = diff(f1, c_var)
f = -c*grad(mu)

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

mu_mid = (1-theta)*mu0 + theta*mu
c_mid = (1-theta)*c0 + theta*c

def L(vm):
    L0 = inner(c - c0, q)*dx + dt*inner(dot(y0, grad(c_mid)), q)*dx + \
         Pe*dt*inner(grad(mu_mid), grad(q))*dx - Pe*dt*vm*q*ds(2)
    LL1 = mu*v*dx - dfdc*Cn*v*dx - Ch*dot(grad(v), grad(c))*dx + Wetting*v*ds(1)
    return L0 + LL1

F1 = (1/k)*inner(y - y0, w)*dx + inner(grad(y0)*y0, w)*dx + \
     Re*inner(2*epsilon(y), epsilon(w))*dx - We*inner(f, w)*dx

a1 = lhs(F1)
L1 = rhs(F1)

a2 = inner(grad(p), grad(r))*dx
L2 = -(1/k)*div(u1)*r*dx

a3 = inner(y, w)*dx
L3 = inner(u1, w)*dx - k*inner(grad(p1), w)*dx

class CahnHilliardEquation1(NonlinearProblem):
    def __init__(self, a, L):
        super().__init__()
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

def Evaporate(sdI):
    vm = Constant(sdI)
    a_form = derivative(L(vm), u, du)   # Jacobian form
    L_form = L(vm)                      # Residual form
    return CahnHilliardEquation1(a_form, L_form)

solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

def droplet_solution(sd, Tfinal, Nt, file_name):
    mfile = File(file_name + "/Potential/result.pvd", "compressed")
    cfile = File(file_name + "/Phasefield/result.pvd", "compressed")
    yfile = File(file_name + "/Velocity/result.pvd", "compressed")
    pfile = File(file_name + "/Pressure/result.pvd", "compressed")
    file = File(file_name + "/final_u_.xml")

    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    it = 0
    t = 0.0
    ts = np.linspace(0, Tfinal, Nt)
    itc = 0

    ctr = -1
    while t < Tfinal:
        ctr +=1
        u0.vector()[:] = u.vector()
        sdI = 0

        solver.solve(Evaporate(sdI), u.vector())

        b1 = assemble(L1)
        for bc in bcu: bc.apply(A1, b1)
        solve(A1, u1.vector(), b1, "gmres", prec)

        b2 = assemble(L2)
        solve(A2, p1.vector(), b2, "gmres", prec)

        b3 = assemble(L3)
        for bc in bcu: bc.apply(A3, b3)
        solve(A3, u1.vector(), b3, "gmres", prec)

        y0.assign(u1)
        it += 1
        t += dt
        
        if ctr % 100 == 0:
            coords = mesh.coordinates()
            phi_vals = u.split()[0].compute_vertex_values(mesh)
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
            cfile << (u.split()[0], t)
            mfile << (u.split()[1], t)
            yfile << u1
            pfile << p1
            itc += 1

    file << u

def main():
    sd = -2
    Tfinal = 1000
    Nsaved = 200
    file_name = "Output/ME_mod"
    droplet_solution(sd, Tfinal, Nsaved, file_name)

if __name__ == "__main__":
    main()
