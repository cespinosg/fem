import numpy as np
import vtk


class Mesh:
    '''
    Creates the mesh where the heat conduction equation will be solved.
    '''

    def __init__(self, nx=3, ny=3):
        self.nx = nx
        self.ny = ny
        self.ne = self.nx*self.ny
        self.np = (self.nx+1)*(self.ny+1)
        self._set_points()
        self._set_connectivity()
        self._calculate_area()

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

    def _set_connectivity(self):
        '''
        Sets the connectivity matrix.
        '''
        self.connectivity = np.zeros((self.ne, 4), dtype=int)
        for i in range(self.nx):
            for j in range(self.ny):
                e = j*self.nx+i
                n = j*(self.nx+1)+i
                self.connectivity[e] = [n, n+1, n+self.nx+2, n+self.nx+1]

    def _calculate_area(self):
        '''
        Calculates the area of all the elements.
        '''
        self.area = np.zeros(self.ne)
        for e in range(self.ne):
            indices = self.connectivity[e]
            points = self.points[:, indices]
            v_1 = points[:, 1]-points[:, 0]
            v_2 = points[:, 2]-points[:, 0]
            v_3 = points[:, 3]-points[:, 0]
            self.area[e] = 0.5*np.linalg.det([v_1, v_2])
            self.area[e] += 0.5*np.linalg.det([v_2, v_3])

    def write(self, vts_file_path):
        '''
        Writes the mesh.
        '''
        # https://examples.vtk.org/site/Cxx/StructuredGrid/StructuredGrid/
        points = vtk.vtkPoints()
        for i in range(self.np):
            points.InsertNextPoint(self.points[0, i], self.points[1, i], 0)
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(self.nx+1, self.ny+1, 1)
        grid.SetPoints(points)
        # https://examples.vtk.org/site/Cxx/PolyData/Casting/
        area = vtk.vtkDoubleArray()
        area.SetNumberOfComponents(1)
        area.SetName('area')
        for e in range(self.ne):
            area.InsertNextValue(self.area[e])
        grid.GetCellData().AddArray(area)
        # https://examples.vtk.org/site/Cxx/IO/XMLStructuredGridWriter/
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(vts_file_path)
        writer.SetInputData(grid)
        writer.Write()


class QuadElement:
    '''
    Calculates the stiffness matrix for a Quadrilateral element.
    '''

    gauss_points = 1/np.sqrt(3)*np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ])
    n_gauss_points = gauss_points.shape[0]
    n_dim = gauss_points.shape[1]

    def __init__(self, points):
        self.points = points
        self.n_points = self.points.shape[1]
        self._calculate_b_xi()
        self._calculate_jacobian()
        self._calculate_area()
        self._calculate_b()
        self._calculate_stiffness()

    def _calculate_b_xi(self):
        '''
        Calculates the B matrix in the element coordinates at the Gauss points.
        '''
        self.b_xi = np.zeros((self.n_gauss_points, self.n_dim, self.n_gauss_points))
        for i in range(self.n_gauss_points):
            xi = self.gauss_points[i, 0]
            eta = self.gauss_points[i, 1]
            self.b_xi[i] = 0.25*np.array([
                [-(1-eta), 1-eta, 1+eta, -(1+eta)],
                [-(1-xi), -(1+xi), 1+xi, 1-xi],
                ])
            # anti-clockwise sorting

    def _calculate_jacobian(self):
        '''
        Calculates the jacobian matrix at the Gauss points.
        '''
        self.jacobian = np.zeros((self.n_gauss_points, self.n_dim, self.n_dim))
        for i in range(self.n_gauss_points):
            self.jacobian[i] = np.dot(self.points, self.b_xi[i].T)

    def _calculate_area(self):
        '''
        Calculates the area.
        '''
        self.area = sum([np.linalg.det(j) for j in self.jacobian])

    def _calculate_b(self):
        '''
        Calculates the B matrix in the physical coordinates at the Gauss points.
        '''
        self.b = np.zeros((self.n_gauss_points, self.n_dim, self.n_gauss_points))
        for i in range(self.n_gauss_points):
            self.b[i] = np.dot(np.linalg.inv(self.jacobian[i]).T, self.b_xi[i])

    def _calculate_stiffness(self):
        '''
        Calculates the stiffness matrix.
        '''
        self.k = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_gauss_points):
            self.k += np.dot(self.b[i].T, self.b[i])*np.linalg.det(self.jacobian[i])


class Assembler:
    '''
    Assembles the element stiffness matrices and the boundary fluxes.
    '''

    def __init__(self, mesh):
        self.mesh = mesh
        self.k = np.zeros((self.mesh.np, self.mesh.np))
        self._calculate_k()

    def _calculate_k(self):
        '''
        Calculates the stiffness matrix.
        '''
        for e in range(self.mesh.ne):
            indices = self.mesh.connectivity[e]
            points = self.mesh.points[:, indices]
            element = QuadElement(points)
            n_points = element.k.shape[0]
            for e_i in range(n_points):
                i = indices[e_i]
                for e_j in range(n_points):
                    j = indices[e_j]
                    self.k[i, j] += element.k[e_i, e_j]


if __name__ == '__main__':
    mesh = Mesh()
    mesh.write('heat_2d.vts')
    # points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    # points = np.array([[0, 0], [1, -1], [2, 0], [1, 1]]).T
    # points = np.array([[0, 0], [3, 0], [2, 1], [1, 1]]).T
    # element = QuadElement(points)
    assembler = Assembler(mesh)

