import numpy as np
import vtk_helper as vh


def lagrange_polynomial(x_nodes, index, x):
    '''
    Evaluates the Lagrange polynomial at the given coordinate.
    '''
    indices = [i for i in range(len(x_nodes)) if i != index]
    l = [(x-x_nodes[i])/(x_nodes[index]-x_nodes[i]) for i in indices]
    return np.prod(l)


def lagrange_polynomial_derivative(x_nodes, index, x):
    '''
    Calculates the derivative of the Lagrange polynomial at the given
    coordinate.
    '''
    indices = [i for i in range(len(x_nodes)) if i != index]
    dl_dx = 0
    for i in indices:
        js = [j for j in indices if j != i]
        l = [(x-x_nodes[j])/(x_nodes[index]-x_nodes[j]) for j in js]
        dl_dx += 1/(x_nodes[index]-x_nodes[i])*np.prod(l)
    return dl_dx


class Element:
    '''
    Base class for elements.
    '''

    def __init__(self, nodes, n_xi=5, n_eta=5, n_zeta=5):
        self.nodes = nodes
        self.n_xi = n_xi
        self.n_eta = n_eta
        self.n_zeta = n_zeta
        self._mesh()

    def _mesh(self):
        '''
        Meshes the element to visualise the mapping.
        '''
        self.n_points = self.n_xi*self.n_eta*self.n_zeta
        self.points = np.zeros((self.n_points, 3))
        self.jacobian = np.zeros(self.n_points)
        self.xi = np.linspace(-1, 1, self.n_xi)
        self.eta = np.linspace(-1, 1, self.n_eta)
        self.zeta = np.linspace(-1, 1, self.n_zeta)
        for self.k in range(self.n_zeta):
            for self.j in range(self.n_eta):
                for self.i in range(self.n_xi):
                    self.p_id = self.j*self.n_xi+self.i
                    self._calculate_coordinates()
                    self._calculate_jacobian()

    def write(self, vts_file_path):
        '''
        Writes the mesh in a vts file.
        '''
        mesh = vh.create_structured_grid(self.points.T, self.n_xi, self.n_eta,
            self.n_zeta)
        vh.add_point_array(mesh, self.jacobian, 'jacobian')
        vh.write_structured_grid(mesh, vts_file_path)


class LinearQuad(Element):
    '''
    Represents a linear quadrilateral element.
    '''

    def _calculate_coordinates(self):
        '''
        Calculates the coordinates for the given xi and eta values.
        '''
        xi = self.xi[self.i]
        eta = self.eta[self.j]
        n = np.array([[
            0.25*(1-xi)*(1-eta),
            0.25*(1+xi)*(1-eta),
            0.25*(1+xi)*(1+eta),
            0.25*(1-xi)*(1+eta),
        ]])
        self.points[self.p_id] = np.dot(n, self.nodes)

    def _calculate_jacobian(self):
        '''
        Calculates the Jacobian at each point of the element mesh.
        '''
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        xi = self.xi[self.i]
        eta = self.eta[self.j]
        j11 = 0.25*(-x[0]+x[1]+x[2]-x[3]+(x[0]-x[1]+x[2]-x[3])*eta)
        j12 = 0.25*(-y[0]+y[1]+y[2]-y[3]+(y[0]-y[1]+y[2]-y[3])*eta)
        j21 = 0.25*(-x[0]-x[1]+x[2]+x[3]+(x[0]-x[1]+x[2]-x[3])*xi)
        j22 = 0.25*(-y[0]-y[1]+y[2]+y[3]+(y[0]-y[1]+y[2]-y[3])*xi)
        self.jacobian[self.p_id] = j11*j22-j21*j12


class Quad(Element):
    '''
    Represents a quadrilateral element of arbitrary order.
    '''

    def _calculate_coordinates(self):
        '''
        Calculates the coordinates.
        '''
        xi = self.xi[self.i]
        eta = self.eta[self.j]
        n = np.array([[self._ni(i, xi, eta) for i in range(self.n_nodes)]])
        self.points[self.p_id] = np.dot(n, self.nodes)

    def _calculate_jacobian(self):
        '''
        Calculates the Jacobian.
        '''
        a = np.zeros((2, self.n_nodes))
        xi = self.xi[self.i]
        eta = self.eta[self.j]
        a[0] = [self._dni_dxi(i, xi, eta) for i in range(self.n_nodes)]
        a[1] = [self._dni_deta(i, xi, eta) for i in range(self.n_nodes)]
        j = np.dot(a, self.nodes)
        self.jacobian[self.p_id] = np.linalg.norm(np.linalg.cross(j[0], j[1]))

    def _ni(self, i, xi, eta):
        '''
        Returns the i-th shape function at the given coordinates.
        '''
        xi_nodes, row_index = self._get_row(i)
        eta_nodes, col_index = self._get_column(i)
        l_row = lagrange_polynomial(xi_nodes, row_index, xi)
        l_col = lagrange_polynomial(eta_nodes, col_index, eta)
        return l_row*l_col

    def _dni_dxi(self, i, xi, eta):
        '''
        Returns the derivative with respect of xi of the i-th shape function at
        the given coordinates.
        '''
        xi_nodes, row_index = self._get_row(i)
        eta_nodes, col_index = self._get_column(i)
        l_row = lagrange_polynomial_derivative(xi_nodes, row_index, xi)
        l_col = lagrange_polynomial(eta_nodes, col_index, eta)
        return l_row*l_col

    def _dni_deta(self, i, xi, eta):
        '''
        Returns the derivative with respect of eta of the i-th shape function at
        the given coordinates.
        '''
        xi_nodes, row_index = self._get_row(i)
        eta_nodes, col_index = self._get_column(i)
        l_row = lagrange_polynomial(xi_nodes, row_index, xi)
        l_col = lagrange_polynomial_derivative(eta_nodes, col_index, eta)
        return l_row*l_col

    def _get_row(self, i):
        '''
        Returns the row where the given node is found.
        '''
        row = [r for r in self.connectivity if i in r][0]
        row_index = [j for j in range(len(row)) if row[j] == i][0]
        xi_nodes = self.xi_nodes[row, 0]
        return xi_nodes, row_index

    def _get_column(self, j):
        '''
        Retuns the column where the given node is found.
        '''
        col = [c for c in self.connectivity.T if j in c][0]
        col_index = [i for i in range(len(col)) if col[i] == j][0]
        eta_nodes = self.xi_nodes[col, 1]
        return eta_nodes, col_index


class Quad2(Quad):
    '''
    Represents a quadrilateral element of order two.
    '''
    
    connectivity = np.array([
        [0, 4, 1],
        [7, 8, 5],
        [3, 6, 2],
    ])
    n_rows, n_cols = connectivity.shape
    xi_nodes = np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, 0],
    ])
    n_nodes = len(xi_nodes)


class Quad1(Quad):
    '''
    Represents a quadrilateral element of order one.
    '''
    
    connectivity = np.array([
        [0, 1],
        [3, 2],
    ])
    n_rows, n_cols = connectivity.shape
    xi_nodes = np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ])
    n_nodes = len(xi_nodes)


if __name__ == '__main__':
    q1_nodes = np.array([
        [0, 0, 0],
        [3, 1, 0],
        [4, 3, 0],
        [1, 2, 0],
    ])
    quad = LinearQuad(q1_nodes, n_xi=3, n_eta=4, n_zeta=1)
    quad.write('results/jacobian/linear-quad.vts')
    q1 = Quad1(q1_nodes, n_xi=3, n_eta=4, n_zeta=1)
    q1.write('results/jacobian/q1.vts')
    q2_nodes = np.array([
        [0, 0, 0],
        [3, 1, 0],
        [4, 3, 0],
        [1, 2, 0],
        [1.5, 0.5, 0],
        [3.5, 2, 0],
        [2.5, 2.5, 0],
        [0.5, 1, 0],
        [2, 1.5, 0],
    ])
    delta = np.sqrt(q2_nodes[3, 0]**2+q2_nodes[3, 1]**2)
    q2_nodes[-1, [0, 1]] += 0.3*delta
    q2 = Quad2(q2_nodes, n_xi=21, n_eta=21, n_zeta=1)
    q2.write('results/jacobian/q2.vts')

