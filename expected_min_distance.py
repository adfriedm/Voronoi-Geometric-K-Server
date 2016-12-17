import numpy as np
import matplotlib.pyplot as plt


def euc_dist(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2), ord=2)

#is P inside triangle made by p1,p2,p3?
def in_triangle(P, p1, p2, p3):
    x,x1,x2,x3 = P[0],p1[0],p2[0],p3[0]
    y,y1,y2,y3 = P[1],p1[1],p2[1],p3[1]
    full = abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    first = abs (x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2))
    second = abs (x1 * (y - y3) + x * (y3 - y1) + x3 * (y1 - y))
    third = abs (x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2))
    return abs(first + second + third - full) < 1e-9


def gen_triangle_pts(p1, p2, p3, n=1e3):
    tri = np.array( [p1,p2,p3], dtype=float)
    x_min, y_min = np.min(tri, axis=0)
    x_max, y_max = np.max(tri, axis=0)
    pts = zip(np.random.uniform(x_min, x_max, n), np.random.uniform(y_min, y_max, n) )
    tr_pts = []
    for pt in pts:
        if in_triangle(pt, p1, p2, p3):
            tr_pts.append( pt )
    return tr_pts


# Generates uniform points from a polygon. Poly is list of coord pairs
def gen_poly_pts(poly, pc, n=1e3):
    poly_arr = np.array( poly, dtype=float)
    x_min, y_min = np.min(poly_arr, axis=0)
    x_max, y_max = np.max(poly_arr, axis=0)
    pts = zip(np.random.uniform(x_min, x_max, n), np.random.uniform(y_min, y_max, n) )
    
    acc_pts = []
    for pt in pts:
        for i in xrange(0,len(poly)):
            if in_triangle(pt, pc, poly[i-1], poly[i]):
                acc_pts.append( pt )
    return acc_pts



# Right angle on CAB. Distance from C
def right_triangle_dint(pa, pb, pc):
    beta = np.linalg.norm(pa-pb)
    gamma = np.linalg.norm(pb-pc)
    
    return np.sqrt(gamma**2-beta**2)*(2*beta*gamma - (gamma**2-beta**2)*np.log((gamma-beta)/(gamma+beta))) / 12.0

#
def triangle_dint(pa, pb, pc, tol=1e-7):
    ts = np.dot(pc-pa, pb-pa) / np.dot(pb-pa, pb-pa)
    pd = ts*pb + (1-ts)*pa
    
    # acute angle
    if ts+tol <= 0.0:
        return right_triangle_dint(pc=pc, pa=pd, pb=pb) - right_triangle_dint(pc=pc, pa=pd, pb=pa)
    # acute angle
    elif ts-tol >= 1.0:
        return right_triangle_dint(pc=pc, pa=pd, pb=pa) - right_triangle_dint(pc=pc, pa=pd, pb=pb)
    # right angle
    elif np.abs(ts) < tol:
        return right_triangle_dint(pc=pc, pa=pa, pb=pb)
    # right angle
    elif np.abs(ts-1.0) < tol:
        return right_triangle_dint(pc=pc, pa=pb, pb=pa)
    # obtuse angle
    else:
        return right_triangle_dint(pc=pc, pa=pd, pb=pa) + right_triangle_dint(pc=pc, pa=pd, pb=pb)

#
def poly_dint(poly, pc, tol=1e-7):
    tot = 0.0
    for i in xrange(0,len(poly)):
        tot += triangle_dint(pa=poly[i-1], pb=poly[i], pc=pc, tol=tol)
    return tot


def main():
    poly = [(1,0), (1.2,1), (0,0.7), (-0.5,0.4), (-0.8,-1.2), (1,-1)]
    poly = [np.array(pt, dtype=float) for pt in poly]
    pc = np.array((0,0), dtype=float)
    
    ### Monte Carlo Approximation of Expectation ###
    # Generate random points from polygon
    pts = gen_poly_pts(poly, pc, n=5e4)
    tot = 0.0
    for pt in pts:
        tot += np.linalg.norm(pt-pc)
    ave = tot / len(pts)
    print ave
    
    ### Explicit Calculation of Expectation ###
    d_int = poly_dint(poly, pc)
    # Now find the area
    def herons(a, b, c): 
        a,b,c = np.sort([a,b,c])
        return 0.25*np.sqrt( (a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)) )
    def poly_area(poly, pc):
        tot = 0.0
        for i in xrange(0,len(poly)):
            a = np.linalg.norm(poly[i-1]-poly[i])
            b = np.linalg.norm(poly[i]-pc)
            c = np.linalg.norm(pc-poly[i-1])
            tot += herons(a, b, c)
        return tot
    print d_int/poly_area(poly, pc)
    
    fig = plt.gcf()
    pts_arr = np.array(pts, dtype=float)
    poly_arr = np.array(poly, dtype=float)
    plt.plot(pts_arr[:,0], pts_arr[:,1], markersize=2, marker='o', color='b', linestyle='None')
    plt.plot(poly_arr[:,0], poly_arr[:,1], marker='o', color='r', linestyle='None')
    plt.plot(pc[0], pc[1], marker='o', color='y', linestyle='None')
    
    plt.axis([-1.5,1.5,-1.5,1.5])
    plt.show()


if __name__ == '__main__':
    main()