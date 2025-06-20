import pdb

import numpy as np

import fem_2d as f2d


def test_quad_element():
    '''
    Tests the implementation of the quadrilateral element.
    '''

    points = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    quad = f2d.QuadElement(points)
    area_check = np.isclose(quad.area, 1)
    print(f'\nArea check: {area_check}')

    diff_func = lambda x, y: 1
    quad.set_diffusivity(diff_func)
    d = quad.calculate_diffusion()
    print(f'\nd = {d}')
    diff_check = np.isclose(d[0, 1], -1/6)
    print(f'\nD_12 check: {diff_check}')

    pe = 100
    local_pe = 0.5*pe
    alpha = 1/np.tanh(local_pe)-1/local_pe
    vel_func = lambda x, y: np.array([pe, 0])
    quad.set_velocity(vel_func)
    c = quad.calculate_convection()
    print(f'\nc = {c}')
    conv_check = np.isclose(c[0, 1], pe*1/6*(1-alpha))
    print(f'\nC_12 check: {conv_check}')

    source_func = lambda x, y: x**2
    quad.set_source(source_func)
    f = quad.calculate_source()
    print(f'\nf = {f}')
    # source_check = np.isclose(f[0], 0.25*(1-alpha))
    # source_check = np.isclose(f[0], 0.25*(1/3-0.5*alpha))
    source_check = np.isclose(f[0], 1/12*(0.5-alpha))
    print(f'\nf_1 check: {source_check}')

    return all([area_check, diff_check, conv_check, source_check])


if __name__ == '__main__':
    test_quad_element()

