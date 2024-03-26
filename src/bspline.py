"""
Bspline file to generate a bspline curve from a set of control points
 - Created by: Francisco Fonseca on March 2024
"""
import numpy as np
import scipy.interpolate as si

def bspline(cv, n=100, degree=3):
    """ 
    Calculate n samples on a bspline

    Inputs:
        cv :      Array of control vertices
        n  :      Number of samples to return, default: 100
        degree:   Curve degree, default: 3
    Output:
        An array of n samples on the curve
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Prevent degree from exceeding count-1, otherwise splev will crash
    degree = np.clip(degree,1,count-1)

    # Calculate knot vector
    kv = np.array([0]*degree + list(range(count-degree+1)) + [count-degree]*degree,dtype='int')

    # Calculate query range
    u = np.linspace(0,(count-degree),n)

    # Calculate and return the result
    return np.array(si.splev(u, (kv,cv.T,degree))).T



if __name__ == "__main__":
    # Example of Boustrophedon path using bspline
    import matplotlib.pyplot as plt
    colors = ('b', 'g', 'g', 'c', 'm', 'y', 'k')

    cv = np.array([
    [ 0.5,  0.5],
    [ 0.5, 39.5],
    [ 5.,  39.5],
    [ 5.,  0.5],
    [ 10.,  0.5],
    [ 10., 39.5],
    [ 15., 39.5],
    [ 15.,  0.5],
    [ 20.,  0.5],
    [ 20., 39.5],
    [ 25., 39.5],
    [ 25.,  0.5],
    [ 30.,  0.5],
    [ 30., 39.5],
    [ 35., 39.5],
    [ 35.,  0.5],
    [ 39.5,  0.5],
    [ 39.5, 39.5]
    ])

    plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')

    for d in range(2,3):
        p = bspline(cv,n=1000,degree=d)
        x,y = p.T
        plt.plot(x,y,'k-',label='Degree %s'%d,color=colors[d%len(colors)])
        # starting from the first waypoint select points that are at approx 5 in length
        # of distance. Consider the first point as the starting point and test the 
        # distance from the next point. If the distance is greater than 5, then consider
        # the next point as the next waypoint. If smaller, then continue to the next point
        # until the distance is greater than 5. Then consider that point as the next waypoint
        p_way = [p[0]]
        for i in range(1, len(p)):
            if np.linalg.norm(p[i] - p_way[-1]) > 2.5:
                p_way.append(p[i])
        p_way = np.array(p_way)
        plt.plot(p_way[:,0], p_way[:,1], 'ro', label='Waypoints')


    p_lenght = 0
    for i in range(len(p)-1):
        p_lenght += np.linalg.norm(p[i+1] - p[i])
    print(p_lenght)
    print(p)
    plt.minorticks_on()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()