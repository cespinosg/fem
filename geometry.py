import numpy as np


def intersection(p1, p2, c, u):
    '''
    Finds the intersection of the givne line with the line defined by the
    centre and the vector.
    '''
    v = p2-p1
    a = np.zeros((2, 2))
    a[:, 0] = u
    a[:, 1] = -v
    b = p1-c
    if np.linalg.det(a) == 0:
        distance = 1e12
    else:
        alpha, beta = np.linalg.solve(a, b)
        p = c+alpha*u
        distance = np.sqrt(sum((p-c)**2))
    return distance


def verify_intersection():
    '''
    Verifies that the intersection implementation is right.
    '''
    p1 = np.array([0, 1])
    p2 = np.array([1, 1])
    c = np.array([0, 0])
    theta = np.linspace(np.radians(1), np.radians(179))
    d1 = 1/np.sin(theta)
    d2 = np.zeros(len(theta))
    for i in range(len(theta)):
        u = np.array([np.cos(theta[i]), np.sin(theta[i])])
        d2[i] = intersection(p1, p2, c, u)
    return np.allclose(d1, d2)


def size(points, direction):
    '''
    Calculates the element size in the given direction.
    '''
    n_points = points.shape[1]
    distance = np.zeros(n_points)
    c = 1/n_points*np.sum(points, axis=1)
    for i in range(n_points-1):
        distance[i] = intersection(points[:, i], points[:, i+1], c, direction)
    distance[-1] = intersection(points[:, 0], points[:, -1], c, direction)
    return sum(np.sort(distance)[:2])


if __name__ == '__main__':
    points = np.array([
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ])
    u = np.array([1, 2])
    print(size(points, u))
    print(verify_intersection())

