import datetime
import pathlib
import time

import numpy as np

import vtk_helper as vh
import geometry as geo


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
        self._set_boundaries()
        self._calculate_area()

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

    def diff_func(self, x, y):
        '''
        Returns the diffusivity at the given coordinates.
        '''
        return 1

    def vel_func(self, x, y):
        '''
        Returns the velocity at the given coordinates.
        '''
        return np.array([1, 0])

    def source_func(self, x, y):
        '''
        Returns the source at the given coordinates.
        '''
        return 0

    def density_func(self, x, y):
        '''
        Returns the density at the given coordinates.
        '''
        return 1

    def check_connectivity(self):
        '''
        Prints the coordinates of each element.
        '''
        for e in range(self.ne):
            print(f'\nElement {e}\n')
            print(f'{self.points[:, self.connectivity[e]]}')

    def write(self, vts_file_path):
        '''
        Writes the mesh.
        '''
        grid = vh.create_structured_grid(self.points, self.nx+1, self.ny+1, 1)
        vh.add_cell_array(grid, self.area, 'area')
        vh.write_structured_grid(grid, vts_file_path)

    def write_boundaries(self, vtm_file_path):
        '''
        Writes the boundaries to the given vtm file path.
        '''
        boundaries = [0 for i in range(len(self.boundaries))]
        names = [name for name in self.boundaries.keys()]
        for (i, b) in enumerate(self.boundaries.values()):
            points = self.points[:, b['nodes']]
            nx = len(b['nodes'])
            boundaries[i] = vh.create_structured_grid(points, nx, 1, 1)
        boundaries = vh.create_multi_block(boundaries, names)
        vh.write_multi_block(boundaries, vtm_file_path)

    def get_phi_0(self):
        '''
        Returns the initial field.
        '''
        if hasattr(self, 'phi_0'):
            return self.phi_0
        else:
            return np.zeros(self.np)

    def set_phi_0_from_file(self, vts_file_path):
        '''
        Sets the initial field value from the given vts file.
        '''
        self.phi_0 = vh.read_phi_from_vts_file(vts_file_path)


class QuadElement:
    '''
    Defines a quadrilateral element.
    '''

    gauss_points_xi = 1/np.sqrt(3)*np.array([
        [-1, 1, 1, -1],
        [-1, -1, 1, 1],
    ])
    n_dim = gauss_points_xi.shape[0]
    n_gauss = gauss_points_xi.shape[1]

    def __init__(self, points):
        self.points = points
        self.n_points = points.shape[1]
        self._calculate_shape_functions()
        self._calculate_gauss_point_coordinates()
        self._calculate_b_xi()
        self._calculate_jacobian()
        self._calculate_area()
        self._calculate_b()

    def _calculate_shape_functions(self):
        '''
        Calculates the shape functions values at the Gauss points.
        '''
        self.n = np.zeros((self.n_gauss, self.n_points))
        for i in range(self.n_gauss):
            xi, eta = self.gauss_points_xi[:, i]
            self.n[i] = [0.25*(1-xi)*(1-eta), 0.25*(1+xi)*(1-eta),
                0.25*(1+xi)*(1+eta), 0.25*(1-xi)*(1+eta)]

    def _calculate_gauss_point_coordinates(self):
        '''
        Calculates the coordinates of the Gauss points.
        '''
        self.gauss_points = np.zeros((self.n_dim, self.n_gauss))
        for i in range(self.n_gauss):
            self.gauss_points[:, i:i+1] = np.dot(self.points, self.n[i, None].T)

    def _calculate_b_xi(self):
        '''
        Calculates the B matrix in the element coordinates at the Gauss points.
        '''
        self.b_xi = np.zeros((self.n_gauss, self.n_dim, self.n_points))
        for i in range(self.n_gauss):
            xi, eta = self.gauss_points_xi[:, i]
            self.b_xi[i] = 0.25*np.array([
                [-(1-eta), 1-eta, 1+eta, -(1+eta)],
                [-(1-xi), -(1+xi), 1+xi, 1-xi],
                ])
            # anti-clockwise sorting

    def _calculate_jacobian(self):
        '''
        Calculates the jacobian matrix at the Gauss points.
        '''
        self.jacobian = np.zeros((self.n_gauss, self.n_dim, self.n_dim))
        self.det_j = np.zeros(self.n_gauss)
        for i in range(self.n_gauss):
            self.jacobian[i] = np.dot(self.points, self.b_xi[i].T)
            self.det_j[i] = np.linalg.det(self.jacobian[i])

    def _calculate_area(self):
        '''
        Calculates the area.
        '''
        self.area = sum(self.det_j)

    def _calculate_b(self):
        '''
        Calculates the B matrix in the physical coordinates at the Gauss points.
        '''
        self.b = np.zeros((self.n_gauss, self.n_dim, self.n_points))
        for i in range(self.n_gauss):
            self.b[i] = np.dot(np.linalg.inv(self.jacobian[i]).T, self.b_xi[i])

    def set_diffusivity(self, diff_func):
        '''
        Sets the diffusivity at the Gauss points.
        '''
        self.k = np.array([diff_func(*g) for g in self.gauss_points.T])

    def calculate_diffusion(self):
        '''
        Calculates the diffusion matrix.
        '''
        d = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_gauss):
            d += self.k[i]*np.dot(self.b[i].T, self.b[i])*self.det_j[i]
        return d

    def set_velocity(self, vel_func):
        '''
        Sets the velocity at the Gauss points.
        '''
        self.u = np.array([vel_func(*g) for g in self.gauss_points.T]).T
        self.mag_u = np.sqrt(np.sum(self.u**2, axis=0))
        self._calculate_convective_derivative()
        self._calculate_weighting_function()

    def _calculate_convective_derivative(self):
        '''
        Calculates the convective derivative at the Gauss points.
        '''
        self.conv_der = np.zeros((self.n_gauss, self.n_points))
        for i in range(self.n_gauss):
            if self.mag_u[i] != 0:
                self.conv_der[i] = np.dot(self.u[:, i].T, self.b[i])

    def _calculate_weighting_function(self):
        '''
        Calculates the weighting function at the Gauss points.
        '''
        self.w = np.zeros((self.n_gauss, self.n_points))
        h = self._calculate_h()
        for i in range(self.n_gauss):
            n = self.n[i, None]
            if self.mag_u[i] != 0:
                pe = 0.5*self.mag_u[i]*h/self.k[i]
                alpha = 1/np.tanh(pe)-1/pe
                self.w[i] = n+alpha*0.5*h*self.conv_der[i]/self.mag_u[i]
            else:
                self.w[i] = n

    def _calculate_h(self):
        '''
        Calculates the element size.
        '''
        mean_u = np.sum(self.u, axis=1)/self.area
        direction = np.array([-mean_u[1], mean_u[0]])
        return geo.size(self.points, direction)

    def calculate_convection(self):
        '''
        Calculates the convection matrix.
        '''
        c = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_gauss):
            c += np.dot(self.w[i, None].T, self.conv_der[i, None])*self.det_j[i]
        return c

    def set_source(self, source_func):
        '''
        Sets the source term at the Gauss quadrature points.
        '''
        self.q = np.array([source_func(*g) for g in self.gauss_points.T])

    def calculate_source(self):
        '''
        Calculates the source term forcing.
        '''
        f = np.zeros(self.n_points)
        for i in range(self.n_gauss):
            f += self.w[i]*self.q[i]*self.det_j[i]
        return f

    def set_density(self, density_func):
        '''
        Sets the density at the Gauss points.
        '''
        self.rho = np.array([density_func(*g) for g in self.gauss_points.T])

    def calculate_mass(self):
        '''
        Calculates the mass matrix.
        '''
        m = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_gauss):
            m += self.rho[i]*np.dot(self.w[i, None].T, self.n[i, None])*self.det_j[i]
        return m


class NeumannBoundary:
    '''
    Calculates the heat flux in an element with a Neumann boundary condition.
    '''

    def __init__(self, points, q):
        self.points = points
        self.q = q
        self._calculate_flux()

    def _calculate_flux(self):
        '''
        Returns the heat flux.
        '''
        dx = self.points[0, 1]-self.points[0, 0]
        dy = self.points[1, 1]-self.points[1, 0]
        ds = np.sqrt(dx**2+dy**2)
        a = np.array([[2, 1], [1, 2]])
        self.f = ds/6*np.dot(a, self.q)


class Solver:
    '''
    Assembles and solves the element stiffness matrices and the boundary fluxes.
    '''

    def __init__(self, mesh):
        self.mesh = mesh
        self.m = np.zeros((self.mesh.np, self.mesh.np))
        self.k = np.zeros((self.mesh.np, self.mesh.np))
        self.f = np.zeros(self.mesh.np)
        self.phi = self.mesh.get_phi_0()

    def _assemble(self):
        '''
        Assembles the stiffness and force matrices.
        '''
        for e in range(self.mesh.ne):
            indices = self.mesh.connectivity[e]
            points = self.mesh.points[:, indices]
            element = QuadElement(points)
            element.set_diffusivity(self.mesh.diff_func)
            k = element.calculate_diffusion()
            element.set_velocity(self.mesh.vel_func)
            c = element.calculate_convection()
            self.k[np.ix_(indices, indices)] += k+c
            element.set_source(self.mesh.source_func)
            self.f[indices] += element.calculate_source()
            element.set_density(self.mesh.density_func)
            self.m[np.ix_(indices, indices)] += element.calculate_mass()

    def _apply_boundary_conditions(self):
        '''
        Applies the boundary conditions.
        '''
        self.dirichlet_nodes = []
        for boundary in self.mesh.boundaries.values():
            if boundary['type'] == 'neumann':
                elements = boundary['connectivity']
                for e in elements:
                    indices = boundary['nodes'][e]
                    points = self.mesh.points[:, indices]
                    q = boundary['values'][e]
                    bc = NeumannBoundary(points, q)
                    self.f[indices] += bc.f
            if boundary['type'] == 'dirichlet':
                self.phi[boundary['nodes']] = boundary['values']
                self.dirichlet_nodes.extend(boundary['nodes'])
        self.unknown_nodes = [i for i in range(self.mesh.np) if i not in self.dirichlet_nodes]

    def write(self, vts_file_path):
        '''
        Writes the results to the given vts file path.
        '''
        grid = vh.create_structured_grid(self.mesh.points, self.mesh.nx+1,
            self.mesh.ny+1, 1)
        vh.add_point_array(grid, self.phi, 'phi')
        vh.write_structured_grid(grid, vts_file_path)


class SteadySolver(Solver):
    '''
    Solves the steady state convection diffusion equation in 2 dimensions.
    '''

    def __init__(self, mesh):
        super().__init__(mesh)
        self._assemble()
        self._apply_boundary_conditions()
        self._solve()

    def _solve(self):
        '''
        Solves the temperature at the nodes where it is not known.
        '''
        k_ll = self.k[np.ix_(self.unknown_nodes, self.unknown_nodes)]
        k_lr = self.k[np.ix_(self.unknown_nodes, self.dirichlet_nodes)]
        f_l = self.f[self.unknown_nodes]
        loading = f_l-np.dot(k_lr, self.phi[self.dirichlet_nodes])
        self.phi[self.unknown_nodes] = np.linalg.solve(k_ll, loading)


class TransientSolver(Solver):
    '''
    Solves the transient convection diffusion equation in 2 dimensions.
    '''

    def __init__(self, mesh, t, write_interval, tol=1e-6):
        super().__init__(mesh)
        self.t = t
        self.write_interval = write_interval
        self.tol = tol
        self._set_log_file()
        self._assemble()
        self._apply_boundary_conditions()
        self._solve()

    def _set_log_file(self):
        '''
        Sets the log file path.
        '''
        self.log_file = pathlib.Path(f'{self.mesh.folder}/{self.mesh.name}.log')
        if not self.log_file.parent.is_dir():
            self.log_file.parent.mkdir(parents=True)
        if self.log_file.is_file():
            self.log_file.unlink()

    def _log(self, message):
        '''
        Prints the given message to the terminal and writes it to the log file.
        '''
        print(message)
        with open(self.log_file, 'a') as fout:
            fout.write(message+'\n')

    def _solve(self):
        '''
        Integrates the convection diffusion equation in time with the Crank-
        Nicolson method.
        '''
        self.nt = len(self.t)
        dt = np.diff(self.t)
        self.diff = np.zeros(self.nt-1)
        m_ll = self.m[np.ix_(self.unknown_nodes, self.unknown_nodes)]
        inv_m_ll = np.linalg.inv(m_ll)
        k_l = self.k[self.unknown_nodes]
        f_l = self.f[self.unknown_nodes]
        self._write_times()
        self._log(f'Starting the simulation at {datetime.datetime.now()}')
        for i in range(self.nt-1):
            new_phi = self.phi.copy()
            previous_new_phi = new_phi.copy()
            error = self.tol+1
            j = 0
            t0 = time.perf_counter()
            while error > self.tol and j < 100:
                r = -np.dot(k_l, 0.5*(self.phi+new_phi))+f_l
                dphi_dt = np.dot(inv_m_ll, r)
                new_phi[self.unknown_nodes] = self.phi[self.unknown_nodes]
                new_phi[self.unknown_nodes] += dphi_dt*dt[i]
                error = max(abs(new_phi-previous_new_phi))
                previous_new_phi = new_phi.copy()
                j += 1
            dt_wall = time.perf_counter()-t0
            self.diff[i] = max(abs(self.phi-new_phi))
            self.phi = new_phi
            ln = (f'Solving for t = {self.t[i+1]:.4f} [-], '
                f'diff = {self.diff[i]:.4e}, j = {j}, error = {error:.4e}, '
                f'dt_wall = {dt_wall:.4e} [s]')
            self._log(ln)
            if (i+1) % self.write_interval == 0 or i == self.nt-2:
                vts_fp = f'{self.mesh.folder}/{self.mesh.name}-{i+1:03}.vts'
                self.write(vts_fp)
        self._log(f'Finishing the simulation at {datetime.datetime.now()}')
            

    def _write_times(self):
        '''
        Writes the times at which the solution is written.
        '''
        tw = [self.t[i] for i in range(self.nt-1) if i % self.write_interval == 0]
        with open(f'{self.mesh.folder}/time.txt', 'w') as fout:
            fout.write('\n'.join([f'{t}' for t in tw]))

