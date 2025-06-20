import itertools as it

import matplotlib.pyplot as plt
import matplotlib.markers as mk
import numpy as np
import scipy.optimize as opt


plt.rcdefaults()
plt.rc('lines', markersize=6)
plt.rc('markers', fillstyle='none')
plt.rc('axes', grid=True)
plt.rc('legend', framealpha=0.7)
plt.rc('savefig', dpi=200, bbox='tight')
plt.rc('figure.constrained_layout', use=True)
plt.close('all')


def get_markers_iterator():
    '''
    Returns a markers iterator.
    '''
    return it.cycle(mk.MarkerStyle.filled_markers)


class Case1:
    '''
    Defines a 1D steady state convection diffusion case with no source term. See
    Figure 2.3 of Zienkiewicz FEM for Fluids (7th edition).
    '''

    def __init__(self, n_elem, phi_0, phi_1, peclet):
        self.n_elem = n_elem
        self.phi_0 = phi_0
        self.phi_1 = phi_1
        self.peclet = peclet
        self._create_mesh()
        self._calculate_analytical_solution()

    def _create_mesh(self):
        '''
        Creates the mesh.
        '''
        self.x = np.linspace(0, 1, self.n_elem+1)
        self.h = np.diff(self.x)

    def _calculate_analytical_solution(self):
        '''
        Calculates the analytical solution.
        '''
        self.x_an = np.linspace(0, 1, 101)
        if self.phi_0 == 0:
            self.phi_an = (np.exp(self.peclet*self.x_an)-1)
            self.phi_an /= (np.exp(self.peclet)-1)
        else:
            self.phi_an = (np.exp(self.peclet*self.x_an)-np.exp(self.peclet))
            self.phi_an /= (1-np.exp(self.peclet))

    def get_velocity(self, x):
        '''
        Returns the velocity at the given location.
        '''
        return self.peclet

    def get_diffusivity(self, x):
        '''
        Returns the diffusivity at the given location.
        '''
        return 1

    def get_source(self, x):
        '''
        Returns the source term at the given location.
        '''
        return 0

    def compare(self, solutions):
        '''
        Compares the analytical with the given numerical solutions.
        '''
        fig, ax = plt.subplots()
        ax.plot(self.x_an, self.phi_an, label='Analytical solution')
        markers = get_markers_iterator()
        for (label, phi) in solutions.items():
            ax.plot(self.x, phi, label=label,
                marker=next(markers))
        ax.legend()
        ax.set_xlabel(r'$\bar{x} = x/L$ [-]')
        ax.set_ylabel('Phi [-]')
        if hasattr(self, 'peclet'):
            ax.set_title(f'Peclet = {self.peclet} [-]')
        fig.show()


class Case2(Case1):
    '''
    Defines a 1D steady state convection diffusion case with a variable source
    term. See Figure 2.6a of Zienkiewicz FEM for Fluids (7th edition).
    '''

    def __init__(self, n_elem, delta_x0):
        self.n_elem = n_elem
        self.delta_x0 = delta_x0
        self.phi_0 = 0
        self.phi_1 = 0
        self.peclet = 200
        self._create_mesh()
        self._calculate_analytical_solution()

    def _create_mesh(self):
        '''
        Creates the mesh.
        '''
        f = lambda g: 1-(g**self.n_elem-1)/(g-1)*self.delta_x0
        self.growth_ratio = opt.brentq(f, 0, 2)
        delta_x = lambda i: self.delta_x0*self.growth_ratio**(self.n_elem-1-i)
        self.h = [delta_x(i) for i in range(self.n_elem)]
        self.x = [sum(self.h[:i]) for i in range(self.n_elem)]+[1]

    def _calculate_analytical_solution(self):
        '''
        Calculates the analytical solution.
        '''
        self.x_an = np.linspace(0, 1, 101)
        a = [[1, 1], [1, np.exp(200)]]
        b = [0, -1/600-1/4e4-1/4e6]
        c1, c2 = np.linalg.solve(a, b)
        self.phi_an = c1+c2*np.exp(200*self.x_an)
        self.phi_an += 1/600*self.x_an**3+1/4e4*self.x_an**2+1/4e6*self.x_an

    def get_source(self, x):
        '''
        Returns the source term at the given location.
        '''
        return x**2


class Case3(Case1):
    '''
    Defines a 1D steady state convection diffusion case with variable source
    and variable velocity. See Figure 2.6b of Ziekiewicz FEM for Fluids (7th
    edition).
    '''

    def __init__(self, n_elem):
        self.n_elem = n_elem
        self.phi_0 = 1
        self.phi_1 = 0
        self._create_mesh()
        self._calculate_analytical_solution()

    def _create_mesh(self):
        '''
        Creates the mesh.
        '''
        self.x = np.linspace(1, 2, self.n_elem+1)
        self.h = np.diff(self.x)

    def _calculate_analytical_solution(self):
        '''
        Calculates the analytical solution.
        '''
        self.x_an = np.linspace(1, 2, 101)
        a = [[1/61, 1], [2**61/61, 1]]
        b = [1-1/228, -1/228*2**4]
        c1, c2 = np.linalg.solve(a, b)
        self.phi_an = c1/61*self.x_an**61+c2+1/228*self.x_an**4

    def get_velocity(self, x):
        '''
        Returns the velocity at the given location.
        '''
        return 60/x

    def get_source(self, x):
        '''
        Returns the source term at the given location.
        '''
        return x**2


class Solver:
    '''
    Solves the steady state 1D convection diffusion using the Petrov-Galerkin
    method.
    '''

    def __init__(self, case):
        self.case = case
        self._set_phi()
        self._assemble()
        self._solve()

    def _set_phi(self):
        '''
        Sets the array of phi, that is, the advected variable.
        '''
        self.phi = np.zeros(self.case.n_elem+1)
        self.phi[0] = self.case.phi_0
        self.phi[-1] = self.case.phi_1

    def _assemble(self):
        '''
        Assembles the global stiffness matrix.
        '''
        self.c = np.zeros((self.case.n_elem+1, self.case.n_elem+1))
        self.num_d = np.zeros((self.case.n_elem+1, self.case.n_elem+1))
        self.d = np.zeros((self.case.n_elem+1, self.case.n_elem+1))
        self.f = np.zeros(self.case.n_elem+1)
        for i in range(self.case.n_elem):
            element_matrices = self._calculate_element_stiffness(i)
            indices = np.ix_([i, i+1], [i, i+1])
            self.c[indices] += element_matrices['c']
            self.num_d[indices] += element_matrices['num_d']
            self.d[indices] += element_matrices['d']
            self.f[[i, i+1]] += element_matrices['f']
        self.k = self.c+self.num_d+self.d

    def _calculate_element_stiffness(self, e):
        '''
        Calculates the element stiffness matrix.
        '''
        self._set_element_functions(e)
        c = self._calculate_convection()
        num_d = self._calculate_numerical_diffusion()
        d = self._calculate_diffusion()
        f = self._calculate_loading()
        return {'c': c, 'num_d': num_d, 'd': d, 'f': f}

    def _set_element_functions(self, e):
        '''
        Defines the element functions needed for calculating the stiffness
        and loading matrices.
        '''
        x1 = self.case.x[e]
        x2 = self.case.x[e+1]
        self.x = lambda xi: 0.5*(x2-x1)*(xi+1)+x1
        self.n1 = lambda xi: 1-(self.x(xi)-x1)/self.case.h[e]
        self.n2 = lambda xi: (self.x(xi)-x1)/self.case.h[e]
        self.dn1_dx = lambda xi: -1/self.case.h[e]
        self.dn2_dx = lambda xi: 1/self.case.h[e]
        self.u = lambda xi: self.case.get_velocity(self.x(xi))
        self.ke = lambda xi: self.case.get_diffusivity(self.x(xi))
        self.peclet = lambda xi: 0.5*self.u(xi)*self.case.h[e]/self.ke(xi)
        self.alpha = lambda xi: 1/np.tanh(self.peclet(xi))-1/self.peclet(xi)
        self.w = lambda xi: 0.5*self.case.h[e]*self.u(xi)/abs(self.u(xi))
        self.dx_dxi = lambda xi: 0.5*self.case.h[e]
        self.fe = lambda xi: self.case.get_source(self.x(xi))

    def _calculate_convection(self):
        '''
        Calculates the convection matrix for the given element.
        '''
        shared = lambda xi: self.u(xi)*self.dx_dxi(xi)
        c11 = lambda xi: shared(xi)*self.n1(xi)*self.dn1_dx(xi)
        c12 = lambda xi: shared(xi)*self.n1(xi)*self.dn2_dx(xi)
        c21 = lambda xi: shared(xi)*self.n2(xi)*self.dn1_dx(xi)
        c22 = lambda xi: shared(xi)*self.n2(xi)*self.dn2_dx(xi)
        c = [[c11, c12], [c21, c22]]
        c = np.array([[self._gauss_quadrature(cij) for cij in ci] for ci in c])
        return c

    def _calculate_numerical_diffusion(self):
        '''
        Calculates the numerical diffusion needed to stabilise the solution.
        '''
        shared = lambda xi: self.alpha(xi)*self.w(xi)*self.u(xi)*self.dx_dxi(xi)
        d11 = lambda xi: shared(xi)*self.dn1_dx(xi)*self.dn1_dx(xi)
        d12 = lambda xi: shared(xi)*self.dn1_dx(xi)*self.dn2_dx(xi)
        d21 = lambda xi: shared(xi)*self.dn2_dx(xi)*self.dn1_dx(xi)
        d22 = lambda xi: shared(xi)*self.dn2_dx(xi)*self.dn2_dx(xi)
        d = [[d11, d12], [d21, d22]]
        d = np.array([[self._gauss_quadrature(dij) for dij in di] for di in d])
        return d

    def _calculate_diffusion(self):
        '''
        Calculates the diffusion matrix for the given element.
        '''
        shared = lambda xi: self.ke(xi)*self.dx_dxi(xi)
        d11 = lambda xi: shared(xi)*self.dn1_dx(xi)*self.dn1_dx(xi)
        d12 = lambda xi: shared(xi)*self.dn1_dx(xi)*self.dn2_dx(xi)
        d21 = lambda xi: shared(xi)*self.dn2_dx(xi)*self.dn1_dx(xi)
        d22 = lambda xi: shared(xi)*self.dn2_dx(xi)*self.dn2_dx(xi)
        d = [[d11, d12], [d21, d22]]
        d = np.array([[self._gauss_quadrature(dij) for dij in di] for di in d])
        return d

    def _calculate_loading(self):
        '''
        Calculates the loading caused by the source term.
        '''
        w1 = lambda xi: self.n1(xi)+self.alpha(xi)*self.w(xi)*self.dn1_dx(xi)
        f1 = lambda xi: w1(xi)*self.fe(xi)*self.dx_dxi(xi)
        w2 = lambda xi: self.n2(xi)+self.alpha(xi)*self.w(xi)*self.dn2_dx(xi)
        f2 = lambda xi: w2(xi)*self.fe(xi)*self.dx_dxi(xi)
        f = [f1, f2]
        f = np.array([self._gauss_quadrature(fi) for fi in f])
        return f

    def _gauss_quadrature(self, integrand):
        '''
        Returns the integral of the given integrand using Gauss Legendre
        quadrature.
        '''
        return integrand(-1/np.sqrt(3))+integrand(1/np.sqrt(3))

    def _solve(self):
        '''
        Solves the equations.
        '''
        boundary = [0, -1]
        internal = [i for i in range(1, self.case.n_elem)]
        indices = np.ix_(internal, boundary)
        self.b = -np.dot(self.k[indices], self.phi[boundary])+self.f[internal]
        indices = np.ix_(internal, internal)
        self.a = self.k[indices]
        self.phi[internal] = np.linalg.solve(self.a, self.b)


if __name__ == '__main__':
    # case = Case1(n_elem=5, phi_0=0, phi_1=1, peclet=5)
    # case = Case2(n_elem=10, delta_x0=1e-2)
    case = Case3(n_elem=10)
    solver = Solver(case)
    solutions = {
        'FEM': solver.phi,
    }
    case.compare(solutions)

