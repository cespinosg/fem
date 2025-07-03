import numpy as np

import fem_2d as f2d



class QuadrilateralMesh(f2d.Mesh):
    '''
    Creates a quadrilateral mesh.
    '''

    def _set_points(self):
        '''
        Creates the points array.
        '''
        self.points = np.zeros((2, self.np))
        y_left = np.linspace(0, 1, self.ny+1)
        y_right = np.linspace(0.5, 1, self.ny+1)
        for j in range(self.ny+1):
            indices = slice(j*(self.nx+1), (j+1)*(self.nx+1))
            self.points[0, indices] = np.linspace(0, 2, self.nx+1)
            self.points[1, indices] = (y_right[j]-y_left[j])/2.0*self.points[0, indices]+y_left[j]

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
                'type': 'dirichlet',
                'nodes': np.array([i for i in range(self.nx+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.nx)]),
                'values': np.array([0 for i in range(self.nx+1)]),
            },
            'right':
            {
                'type': 'neumann',
                'nodes': np.array([(i+1)*(self.nx+1)-1 for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'values': np.array([0 for i in range(self.ny+1)]),
            },
            'top':
            {
                'type': 'neumann',
                'nodes': np.array([self.ny*(self.nx+1)+i for i in range(self.nx+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.nx)]),
                'values': np.array([20 for i in range(self.nx+1)]),
            },
        }

    def diff_func(self, x, y):
        '''
        Returns the diffusivity at the given coordinates.
        '''
        return 10

    def vel_func(self, x, y):
        '''
        Returns the velocity at the given coordinates.
        '''
        return np.array([0, 0])


class RectangularMesh(f2d.Mesh):
    '''
    Creates a rectangular mesh.
    '''

    folder = 'results/heat_2d/transient-rectangle'
    name = 'transient-rectangle'

    def _set_points(self):
        '''
        Creates the points array.
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
                'values': np.array([20 for i in range(self.ny+1)]),
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
        Returns the velocity at the given coordinates.
        '''
        return np.array([0, 0])


class CylindricalMesh(f2d.Mesh):
    '''
    Creates a cylindrical mesh.
    '''

    def _set_points(self):
        '''
        Creates the points array.
        '''
        self.points = np.zeros((2, self.np))
        theta = np.linspace(0, 0.5*np.pi, self.nx+1)
        radius = np.linspace(1, 2, self.ny+1)
        for j in range(self.ny+1):
            indices = slice(j*(self.nx+1), (j+1)*(self.nx+1))
            self.points[0, indices] = radius[j]*np.cos(theta)
            self.points[1, indices] = radius[j]*np.sin(theta)

    def _set_boundaries(self):
        '''
        Sets the boundary conditions.
        '''
        self.boundaries = {
            'bottom':
            {
                'type': 'neumann',
                'nodes': np.array([i*(self.nx+1) for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'values': np.array([0 for i in range(self.ny+1)]),
            },
            'left':
            {
                'type': 'dirichlet',
                'nodes': np.array([i for i in range(self.nx+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.nx)]),
                'values': np.array([0 for i in range(self.nx+1)]),
            },
            'top':
            {
                'type': 'neumann',
                'nodes': np.array([(i+1)*(self.nx+1)-1 for i in range(self.ny+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.ny)]),
                'values': np.array([0 for i in range(self.ny+1)]),
            },
            'right':
            {
                'type': 'dirichlet',
                'nodes': np.array([self.ny*(self.nx+1)+i for i in range(self.nx+1)]),
                'connectivity': np.array([[i, i+1] for i in range(self.nx)]),
                'values': np.array([20 for i in range(self.nx+1)]),
            },
        }

    def vel_func(self, x, y):
        '''
        Returns the velocity at the given coordinates.
        '''
        return np.array([0, 0])


class SineWave(f2d.Mesh):
    '''
    Represents the convection diffusion of a sine wave.
    '''

    folder = 'results/heat_2d/transient-sine-wave'
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
        Returns the diffusivity at the given coordinates.
        '''
        return 1e-1

    def vel_func(self, x, y):
        '''
        Returns the velocity at the given coordinates.
        '''
        return np.array([0, 0])

    def set_phi_0(self):
        '''
        Sets the initial value for phi.
        '''
        self.phi_0 = np.zeros(self.np)
        x = self.points[0, :self.nx+1]
        for i in range(self.ny+1):
            offset = i*(self.nx+1)
            self.phi_0[offset:offset+self.nx+1] = np.sin(np.pi*x)


class CosineWave(SineWave):
    '''
    Represents the convection diffusion of a sine wave.
    '''

    folder = 'results/heat_2d/transient-cosine-wave'
    name = 'transient-cosine-wave'

    def set_phi_0(self):
        '''
        Sets the initial value for phi.
        '''
        self.phi_0 = np.zeros(self.np)
        x = self.points[0, :self.nx+1]
        for i in range(self.ny+1):
            offset = i*(self.nx+1)
            self.phi_0[offset:offset+self.nx+1] = np.cos(np.pi*x)


if __name__ == '__main__':
    # mesh = QuadrilateralMesh(nx=10, ny=10)
    mesh = RectangularMesh(nx=10, ny=1)
    # mesh = CylindricalMesh(nx=10, ny=10)
    # mesh.write('results/heat_2d/mesh.vts')
    # mesh.write_boundaries('results/heat_2d/boundaries.vtm')
    # points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    # points = np.array([[0, 0], [1, -1], [2, 0], [1, 1]]).T
    # points = np.array([[0, 0], [3, 0], [2, 1], [1, 1]]).T
    # element = QuadElement(points)
    # mesh = SineWave(nx=50, ny=50)
    mesh = CosineWave(nx=50, ny=50)
    mesh.set_phi_0()
    # solver = f2d.SteadySolver(mesh)
    # solver.write('results/heat_2d/solution.vts')
    solver = f2d.TransientSolver(mesh, np.arange(0, 1.0+1e-4, 1e-4), 100)

