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

    def vel_func(self, x, y):
        '''
        Returns the velocity vector at the given coordinates.
        '''
        return np.array([5, 0])

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

    def check_solution(self, phi):
        '''
        Checks that the analytical solution matches the numerical one.
        '''
        a = [[1, 1], [1, np.exp(200)]]
        b = [0, -1/600-1/4e4-1/4e6]
        c1, c2 = np.linalg.solve(a, b)
        x = self.points[0, :self.nx]
        analytical = c1+c2*np.exp(200*x)+1/600*x**3+1/4e4*x**2+1/4e6*x
        check = np.allclose(phi[:self.nx], analytical, atol=1e-6)
        print(f'Exponential flow with variable source check: {check}')


class SmithHutton(f2d.Mesh):
    '''
    Represents the Smith-Hutton problem.
    '''

    folder = 'results/conv_diff_2d/transient-smith-hutton'
    name = 'transient-smith-hutton'

    def _set_points(self):
        '''
        Sets the point coordinates.
        '''
        x = np.linspace(-1, 1, self.nx+1)
        y = np.linspace(0, 1, self.ny+1)
        self.points = np.array([[xi, yi] for yi in y for xi in x]).T

    def _set_boundaries(self):
        '''
        Sets the boundary conditions.
        '''
        half_nx = int(0.5*self.nx)
        inlet_ids = [i for i in range(half_nx+1)]
        outlet_ids = [i for i in range(half_nx, 2*half_nx+1)]
        x_in = self.points[0, inlet_ids]
        phi_dirichlet = 1-np.tanh(10)
        self.boundaries = {
            'inlet':
            {
                'type': 'dirichlet',
                'nodes': np.array([i for i in inlet_ids]),
                'connectivity': np.array([[i, i+1] for i in range(half_nx)]),
                'values': 1+np.tanh(10*(2*x_in+1)),
            },
            'outlet':
            {
                'type': 'neumann',
                'nodes': np.array([i for i in outlet_ids]),
                'connectivity': np.array([[i, i+1] for i in range(half_nx)]),
                'values': np.zeros(half_nx+1),
            },
            'left':
            {
                'type': 'dirichlet',
                'nodes': np.array([i*(self.nx+1) for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'values': phi_dirichlet*np.ones(self.ny+1),
            },
            'right':
            {
                'type': 'dirichlet',
                'nodes': np.array([(i+1)*(self.nx+1)-1 for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'values': phi_dirichlet*np.ones(self.ny+1),
            },
            'top':
            {
                'type': 'dirichlet',
                'nodes': np.array([self.ny*(self.nx+1)+i for i in range(self.nx+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.nx)]),
                'values': phi_dirichlet*np.ones(self.nx+1),
            },
        }

    def diff_func(self, x, y):
        '''
        Returns the diffusivity at the given cooridinates.
        '''
        return 1e-1

    def vel_func(self, x, y):
        '''
        Returns the velocity vector at the given coordinates.
        '''
        return np.array([2*y*(1-x**2), -2*x*(1-y**2)])


class ManufacturedSolutionDirichlet(f2d.Mesh):
    '''
    Represents the convection-diffusion of a manufactured solution with
    Dirichlet boundary conditions.
    '''

    folder = 'results/conv_diff_2d/transient-ms-dirichlet'
    name = 'transient-ms-dirichlet'
    update_source = True

    def _set_points(self):
        '''
        Sets the point coordinates.
        '''
        x = np.linspace(-1, 1, self.nx+1)
        y = np.linspace(-1, 1, self.ny+1)
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

    def diff_func(self, x, y):
        '''
        Returns the diffusivity at the given cooridinates.
        '''
        return 1e-1

    def vel_func(self, x, y):
        '''
        Returns the velocity vector at the given coordinates.
        '''
        return np.array([1, 0])

    def source_func(self, x, y, t=0):
        '''
        Returns the velocity vector at the given coordinates.
        '''
        alpha = self.diff_func(x, y)
        u = self.vel_func(x, y)
        return (2*alpha-2*u[0]*x-(1-x**2))*np.exp(-t)

    def set_phi_0(self):
        '''
        Sets the initial value for phi.
        '''
        self.phi_0 = np.zeros(self.np)
        x = self.points[0, :self.nx+1]
        for i in range(self.ny+1):
            offset = i*(self.nx+1)
            self.phi_0[offset:offset+self.nx+1] = 1-x**2


class SineWave(f2d.Mesh):
    '''
    Represents the convection diffusion of a sine wave.
    '''

    folder = 'results/conv_diff_2d/transient-sine-wave'
    name = 'transient-sine-wave'

    def _set_points(self):
        '''
        Sets the point coordinates.
        '''
        x = np.linspace(-1, 1, self.nx+1)
        y = np.linspace(-1, 1, self.ny+1)
        self.points = np.array([[xi, yi] for yi in y for xi in x]).T

    def _set_boundaries(self):
        '''
        Sets the boundary conditions.
        '''
        self.boundaries = {
            'left':
            {
                'type': 'periodic',
                'nodes': np.array([i*(self.nx+1) for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'connected_to': 'right',
                'is_solved': True,
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
                'type': 'periodic',
                'nodes': np.array([(i+1)*(self.nx+1)-1 for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'connected_to': 'left',
                'is_solved': False,
            },
            'top':
            {
                'type': 'neumann',
                'nodes': np.array([self.ny*(self.nx+1)+i for i in range(self.nx+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.nx)]),
                'values': np.array([0 for i in range(self.nx+1)]),
            },
        }

    def diff_func(self, x, y):
        '''
        Returns the diffusivity at the given cooridinates.
        '''
        return 1e-1

    def set_phi_0(self):
        '''
        Sets the initial value for phi.
        '''
        self.phi_0 = np.zeros(self.np)
        x = self.points[0, :self.nx+1]
        for i in range(self.ny+1):
            offset = i*(self.nx+1)
            self.phi_0[offset:offset+self.nx+1] = np.sin(np.pi*x)


if __name__ == '__main__':
    # mesh = ExponentialFlow(nx=10, ny=10)
    # mesh = ExpFlowVarSource(nx=30, ny=30)
    mesh = SmithHutton(nx=100, ny=50)
    # mesh = ManufacturedSolutionDirichlet(nx=50, ny=50)
    # mesh = SineWave(nx=50, ny=50)
    # mesh.set_phi_0()
    # solver = f2d.SteadySolver(mesh)
    # solver.write('results/conv_diff_2d/solution.vts')
    # mesh.check_solution(solver.phi)
    solver = f2d.TransientSolver(mesh, np.arange(0, 3.0+1e-4, 1e-4), 100)

