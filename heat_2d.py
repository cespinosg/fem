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

    def _set_points(self):
        '''
        Creates the points array.
        '''
        self.points = np.zeros((2, self.np))
        y_left = np.linspace(0, 1, self.ny+1)
        y_right = np.linspace(0, 1, self.ny+1)
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


if __name__ == '__main__':
    mesh = QuadrilateralMesh(nx=10, ny=10)
    # mesh = CylindricalMesh(nx=10, ny=10)
    # mesh.write('results/heat_2d/mesh.vts')
    # mesh.write_boundaries('results/heat_2d/boundaries.vtm')
    # points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    # points = np.array([[0, 0], [1, -1], [2, 0], [1, 1]]).T
    # points = np.array([[0, 0], [3, 0], [2, 1], [1, 1]]).T
    # element = QuadElement(points)
    solver = f2d.Solver(mesh)
    solver.write('results/heat_2d/solution.vts')

