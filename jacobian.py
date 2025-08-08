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
        self._aux_funcs()
        self._mesh()

    def _aux_funcs(self):
        '''
        Calls auxiliary functions if needed.
        '''
        pass

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
                    self.p_id = self.k*self.n_xi*self.n_eta+self.j*self.n_xi
                    self.p_id += self.i
                    self._calculate_coordinates()
                    self._calculate_jacobian()

    def _get_dimension_array(self, i, dim):
        '''
        Returns the coodinates of the nodes aligned with the given node in the
        given dimension.
        '''
        index = np.where(self.connectivity == i)
        index = [idx[0] for idx in index]
        if len(index) == 2:
            if dim == 0:
                indices = self.connectivity[index[0], :]
                index = index[1]
            if dim == 1:
                indices = self.connectivity[:, index[1]]
                index = index[0]
        elif len(index) == 3:
            if dim == 0:
                indices = self.connectivity[index[0], index[1], :]
                index = index[2]
            if dim == 1:
                indices = self.connectivity[index[0], :, index[2]]
                index = index[1]
            if dim == 2:
                indices = self.connectivity[:, index[1], index[2]]
                index = index[0]
        coords = self.xi_nodes[indices, dim]
        return coords, index

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

    def _aux_funcs(self):
        '''
        Calls the auxiliary functions before creating the mesh.
        '''
        self._calculate_surface_normal()

    def _calculate_surface_normal(self):
        '''
        Calculates the surface normal vector of the element. It is used to
        determine the sign of the Jacobian.
        '''
        u = self.nodes[1]-self.nodes[0]
        v = self.nodes[3]-self.nodes[0]
        self.n = np.cross(u, v)
        self.n = self.n/np.linalg.norm(self.n)

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
        self.jacobian[self.p_id] = np.dot(np.linalg.cross(j[0], j[1]), self.n)

    def _ni(self, i, xi, eta):
        '''
        Returns the i-th shape function at the given coordinates.
        '''
        xi_nodes, xi_index = self._get_dimension_array(i, 0)
        eta_nodes, eta_index = self._get_dimension_array(i, 1)
        l_xi = lagrange_polynomial(xi_nodes, xi_index, xi)
        l_eta = lagrange_polynomial(eta_nodes, eta_index, eta)
        return l_xi*l_eta

    def _dni_dxi(self, i, xi, eta):
        '''
        Returns the derivative with respect of xi of the i-th shape function at
        the given coordinates.
        '''
        xi_nodes, xi_index = self._get_dimension_array(i, 0)
        eta_nodes, eta_index = self._get_dimension_array(i, 1)
        l_xi = lagrange_polynomial_derivative(xi_nodes, xi_index, xi)
        l_eta = lagrange_polynomial(eta_nodes, eta_index, eta)
        return l_xi*l_eta

    def _dni_deta(self, i, xi, eta):
        '''
        Returns the derivative with respect of eta of the i-th shape function at
        the given coordinates.
        '''
        xi_nodes, xi_index = self._get_dimension_array(i, 0)
        eta_nodes, eta_index = self._get_dimension_array(i, 1)
        l_xi = lagrange_polynomial(xi_nodes, xi_index, xi)
        l_eta = lagrange_polynomial_derivative(eta_nodes, eta_index, eta)
        return l_xi*l_eta


class Quad2(Quad):
    '''
    Represents a quadrilateral element of order two.
    '''

    connectivity = np.array([
        [0, 4, 1],
        [7, 8, 5],
        [3, 6, 2],
    ])
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
    xi_nodes = np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ])
    n_nodes = len(xi_nodes)


class Hexa(Element):
    '''
    Represents a hexahedral element.
    '''

    def _calculate_coordinates(self):
        '''
        Calculates the coordinates.
        '''
        xi = self.xi[self.i]
        eta = self.eta[self.j]
        zeta = self.eta[self.k]
        n = np.array([[self._ni(i, xi, eta, zeta) for i in range(self.n_nodes)]])
        self.points[self.p_id] = np.dot(n, self.nodes)

    def _calculate_jacobian(self):
        '''
        Calculates the Jacobian.
        '''
        a = np.zeros((3, self.n_nodes))
        xi_coords = np.array([
            self.xi[self.i], self.eta[self.j], self.zeta[self.k]])
        a[0] = [self._dni_dxi(i, xi_coords, 0) for i in range(self.n_nodes)]
        a[1] = [self._dni_dxi(i, xi_coords, 1) for i in range(self.n_nodes)]
        a[2] = [self._dni_dxi(i, xi_coords, 2) for i in range(self.n_nodes)]
        j = np.dot(a, self.nodes)
        self.jacobian[self.p_id] = np.linalg.det(j)

    def _ni(self, i, xi, eta, zeta):
        '''
        Returns the i-th shape function at the given coordinates.
        '''
        xi_nodes, xi_index = self._get_dimension_array(i, 0)
        eta_nodes, eta_index = self._get_dimension_array(i, 1)
        zeta_nodes, zeta_index = self._get_dimension_array(i, 2)
        l_xi = lagrange_polynomial(xi_nodes, xi_index, xi)
        l_eta = lagrange_polynomial(eta_nodes, eta_index, eta)
        l_zeta = lagrange_polynomial(zeta_nodes, zeta_index, zeta)
        return l_xi*l_eta*l_zeta

    def _dni_dxi(self, i, xi_coords, derivative_dim):
        '''
        Returns the derivative with respect of xi of the i-th shape function at
        the given coordinates.
        '''
        nodes = np.zeros((3, len(self.connectivity)))
        index = np.zeros(3, dtype=int)
        nodes[0], index[0] = self._get_dimension_array(i, 0)
        nodes[1], index[1] = self._get_dimension_array(i, 1)
        nodes[2], index[2] = self._get_dimension_array(i, 2)
        l = 1
        for dim in range(3):
            if dim == derivative_dim:
                li = lagrange_polynomial_derivative(nodes[dim], index[dim],
                    xi_coords[dim])
            else:
                li = lagrange_polynomial(nodes[dim], index[dim], xi_coords[dim])
            l *= li
        return l


class Hexa1(Hexa):
    '''
    Reperesents a hexahedral element of order 1.
    '''
    
    connectivity = np.array([
        [[0, 1],
        [3, 2]],
        
        [[4, 5],
        [7, 6]],
    ])
    xi_nodes = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ])
    n_nodes = len(xi_nodes)


class Hexa2(Hexa):
    '''
    Reperesents a hexahedral element of order 2.
    '''
    
    connectivity = np.array([
        [[0, 8, 1],
        [11, 20, 9],
        [3, 10, 2]],
        
        [[16, 22, 17],
        [24, 26, 25],
        [19, 23, 18]],
        
        [[4, 12, 5],
        [15, 21, 13],
        [7, 14, 6]],
    ])
    xi_nodes = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [0, -1, -1],
        [1, 0, -1],
        [0, 1, -1],
        [-1, 0, -1],
        [0, -1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [-1, 0, 1],
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
        [0, -1, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    n_nodes = len(xi_nodes)

    @staticmethod
    def elevate(nodes):
        '''
        Elevates the given nodes to order 2.
        '''
        n2 = np.zeros((27, 3))
        n2[:8] = nodes[:8]
        n2[8] = 0.5*(n2[0]+n2[1])
        n2[9] = 0.5*(n2[1]+n2[2])
        n2[10] = 0.5*(n2[2]+n2[3])
        n2[11] = 0.5*(n2[3]+n2[0])
        n2[12] = 0.5*(n2[4]+n2[5])
        n2[13] = 0.5*(n2[5]+n2[6])
        n2[14] = 0.5*(n2[6]+n2[7])
        n2[15] = 0.5*(n2[7]+n2[4])
        n2[16] = 0.5*(n2[0]+n2[4])
        n2[17] = 0.5*(n2[1]+n2[5])
        n2[18] = 0.5*(n2[2]+n2[6])
        n2[19] = 0.5*(n2[3]+n2[7])
        n2[20] = 0.5*(n2[8]+n2[10])
        n2[21] = 0.5*(n2[12]+n2[14])
        n2[22] = 0.5*(n2[16]+n2[17])
        n2[23] = 0.5*(n2[19]+n2[18])
        n2[24] = 0.5*(n2[16]+n2[19])
        n2[25] = 0.5*(n2[17]+n2[18])
        n2[26] = 0.5*(n2[22]+n2[23])
        return n2


if __name__ == '__main__':

    q1_nodes = np.array([
        [0, 0, 0],
        [3, 1, 0],
        [4, 3, 0],
        [1, 2, 0],
    ])
    # quad = LinearQuad(q1_nodes, n_xi=3, n_eta=4, n_zeta=1)
    # quad.write('results/jacobian/linear-quad.vts')
    # q1 = Quad1(q1_nodes, n_xi=3, n_eta=4, n_zeta=1)
    # q1.write('results/jacobian/q1.vts')
    
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
    # q2 = Quad2(q2_nodes, n_xi=21, n_eta=21, n_zeta=1)
    # q2.write('results/jacobian/q2.vts')
    
    h1_nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 2, 0],
        [0, 2, 0],
        [0, 0, 3],
        [1, 0, 3],
        [1, 2, 3],
        [0, 2, 3],
    ])
    h1 = Hexa1(h1_nodes)
    h1.write('results/jacobian/h1.vts')
    
    h2_nodes = Hexa2.elevate(h1_nodes)
    delta = np.sqrt(sum(h2_nodes[7]**2))
    h2_nodes[-1] += 0.1*delta
    h2 = Hexa2(h2_nodes, n_xi=21, n_eta=21, n_zeta=21)
    h2.write('results/jacobian/h2.vts')

