import itertools as it

import matplotlib.pyplot as plt
import matplotlib.markers as mk
import numpy as np


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


class Solver:
    '''
    Base class for the solvers.
    '''

    def __init__(self, peclet, n_elem, phi_0, phi_1):
        self.peclet = peclet
        self.n_elem = n_elem
        self.phi_0 = phi_0
        self.phi_1 = phi_1
        self._set_phi()
        self._assemble()
        self._solve()

    def _set_phi(self):
        '''
        Sets the array of phi, that is, the advected variable.
        '''
        self.phi = np.zeros(self.n_elem+1)
        self.phi[0] = self.phi_0
        self.phi[-1] = self.phi_1


class FEMSolver(Solver):
    '''
    Solves the steady state 1D convection diffusion using the Petrov-Galerkin method.
    '''

    def _assemble(self):
        '''
        Assembles the global stiffness matrix.
        '''
        self._calculate_element_stiffness()
        self.k = np.zeros((self.n_elem+1, self.n_elem+1))
        for i in range(self.n_elem):
            indices = np.ix_([i, i+1], [i, i+1])
            self.k[indices] += self.k_e

    def _calculate_element_stiffness(self):
        '''
        Calculates the element stiffness matrix.
        '''
        peclet = self.peclet*0.5/self.n_elem
        print(f'Local Peclet = {peclet:.2f} [-]')
        alpha = 1/np.tanh(peclet)-1/peclet
        # alpha = 1
        k_conv = peclet*np.array([[-1, 1], [-1, 1]])
        k_diff = (1+alpha*peclet)*np.array([[1, -1], [-1, 1]])
        self.k_e = k_conv+k_diff
        # self.k_e = k_diff

    def _solve(self):
        '''
        Solves the equations.
        '''
        boundary = [0, -1]
        internal = [i for i in range(1, self.n_elem)]
        indices = np.ix_(internal, boundary)
        self.b = -np.dot(self.k[indices], self.phi[boundary])
        indices = np.ix_(internal, internal)
        self.a = self.k[indices]
        self.phi[internal] = np.linalg.solve(self.a, self.b)
        self.solution = {'x': np.linspace(0, 1, self.n_elem+1), 'phi': self.phi}


class FVMSolver(Solver):
    '''
    Solves the same equation with the FVM.
    '''

    def _assemble(self):
        '''
        Creates the matrix with the coefficients.
        '''
        self.k = np.zeros((self.n_elem+1, self.n_elem+1))
        for i in range(1, self.n_elem):
            ae = 1/(np.exp(self.peclet/self.n_elem)-1)
            aw = np.exp(self.peclet/self.n_elem)/(np.exp(self.peclet/self.n_elem)-1)
            self.k[i, i-1] = -aw
            self.k[i, i] = ae+aw
            self.k[i, i+1] = -ae

    def _solve(self):
        '''
        Solves the set of equations.
        '''
        boundary = [0, -1]
        internal = [i for i in range(1, self.n_elem)]
        indices = np.ix_(internal, boundary)
        self.b = np.dot(-self.k[indices], self.phi[boundary])
        indices = np.ix_(internal, internal)
        self.a = self.k[indices]
        self.phi[internal] = np.linalg.solve(self.a, self.b)
        self.solution = {'x': np.linspace(0, 1, self.n_elem+1), 'phi': self.phi}


class Comparator:
    '''
    Compares the different solutions with the analytical one.
    '''

    def __init__(self, peclet, phi_0, solutions):
        self.peclet = peclet
        self.phi_0 = phi_0
        self.solutions = solutions
        self._calculate_analytical_solution()
        self._plot()

    def _calculate_analytical_solution(self):
        '''
        Calculates the analytical solution.
        '''
        self.x = np.linspace(0, 1, 101)
        if self.phi_0 == 0:
            self.phi = (np.exp(self.peclet*self.x)-1)/(np.exp(self.peclet)-1)
        else:
            self.phi = (np.exp(self.peclet*self.x)-np.exp(self.peclet))/(1-np.exp(self.peclet))

    def _plot(self):
        '''
        Compares the analytical and numerical solutions.
        '''
        fig, ax = plt.subplots()
        ax.plot(self.x, self.phi, label='Analytical solution')
        markers = get_markers_iterator()
        for (label, solution) in self.solutions.items():
            ax.plot(solution['x'], solution['phi'], label=label,
                marker=next(markers))
        ax.legend()
        ax.set_xlabel(r'$\bar{x} = x/L$ [-]')
        ax.set_ylabel('Phi [-]')
        ax.set_title(f'Peclet = {self.peclet} [-]')
        fig.show()


if __name__ == '__main__':
    n_elem = 9
    phi_0 = 0
    phi_1 = 1
    peclet = 18
    fem_solver = FEMSolver(peclet, n_elem, phi_0, phi_1)
    fvm_solver = FVMSolver(peclet, n_elem, phi_0, phi_1)
    solutions = {'FEM': fem_solver.solution, 'FVM': fvm_solver.solution}
    comparator = Comparator(peclet, phi_0, solutions)

