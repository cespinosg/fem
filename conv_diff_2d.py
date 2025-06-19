import numpy as np

import fem_2d as f2d


class ExponentialFlow(f2d.Mesh):
    '''
    Represents the exponential flow problem.
    '''

    def _set_points(self):
        '''
        Sets the point coordinates.
        '''
        x = np.linspace(0, 1, self.nx+1)
        y = np.linspace(0, 1, self.ny+1)
        self.points = np.array([[xi, yi] for yi in y for xi in x]).T

    def _set_boundaries(self):
        '''
        Sets the boundary conditions.
        '''
        self.boundaries = {
            'left':
            {
                'type': 'dirichlet',
                'nodes': np.array([i*(self.nx+1) for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'values': np.array([0 for i in range(self.ny+1)]),
            },
            'bottom':
            {
                'type': 'neumann',
                'nodes': np.array([i for i in range(self.nx+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.nx)]),
                'values': np.array([0 for i in range(self.nx+1)]),
            },
            'right':
            {
                'type': 'dirichlet',
                'nodes': np.array([(i+1)*(self.nx+1)-1 for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'values': np.array([1 for i in range(self.ny+1)]),
            },
            'top':
            {
                'type': 'neumann',
                'nodes': np.array([self.ny*(self.nx+1)+i for i in range(self.nx+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.nx)]),
                'values': np.array([0 for i in range(self.nx+1)]),
            },
        }

    def check_solution(self, phi):
        '''
        Checks that the numerical solution matches the analytical one.
        '''
        point = self.points[:, 0]
        pe = np.sqrt(sum(self.vel_func(*point)**2))/self.diff_func(*point)
        analytical = (np.exp(pe*self.points[0, :self.nx])-1)/(np.exp(pe)-1)
        check = np.allclose(phi[:self.nx], analytical)
        print(f'Exponential flow check: {check}')


class ExpFlowVarSource(ExponentialFlow):
    '''
    Exponential flow with variable source term.
    '''

    def _set_boundaries(self):
        '''
        Sets the boundary conditions.
        '''
        self.boundaries = {
            'left':
            {
                'type': 'dirichlet',
                'nodes': np.array([i*(self.nx+1) for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'values': np.array([0 for i in range(self.ny+1)]),
            },
            'bottom':
            {
                'type': 'neumann',
                'nodes': np.array([i for i in range(self.nx+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.nx)]),
                'values': np.array([0 for i in range(self.nx+1)]),
            },
            'right':
            {
                'type': 'dirichlet',
                'nodes': np.array([(i+1)*(self.nx+1)-1 for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'values': np.array([0 for i in range(self.ny+1)]),
            },
            'top':
            {
                'type': 'neumann',
                'nodes': np.array([self.ny*(self.nx+1)+i for i in range(self.nx+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.nx)]),
                'values': np.array([0 for i in range(self.nx+1)]),
            },
        }

    def vel_func(self, x, y):
        '''
        Returns the velocity vector at the given coordinates.
        '''
        return np.array([200, 0])

    def source_func(self, x, y):
        '''
        Returns the source at the given coordinates.
        '''
        return x**2


if __name__ == '__main__':
    # mesh = ExponentialFlow(nx=10, ny=10)
    mesh = ExpFlowVarSource(nx=2, ny=2)
    solver = f2d.Solver(mesh)
    solver.write('results/conv_diff_2d/solution.vts')
    # mesh.check_solution(solver.phi)

