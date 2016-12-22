import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial
import scipy.spatial.distance as dist

import sys

import time

eps = sys.float_info.epsilon



def voronoi(points, bounding_box, tol=100*eps):
    # This part was heavily borrowed from an online post that appears
    # to have been deleted. This uses reflective properties of the
    # Voronoi diagrams to place a bounding square. 
    # This ultimately involves copying the same set of points above,
    # below and to each side of the bounding box.
    # In otherwords we are computing the Voronoi on 5 times as many
    # points as we need to. Not good. Very hacky.
    points_arr = np.array(points)
    points_left = np.copy(points_arr)
    points_left[:, 0] = bounding_box[0] \
                      - points_left[:, 0] \
                      - bounding_box[0]
    points_right = np.copy(points_arr)
    points_right[:, 0] = bounding_box[1] \
                        + bounding_box[1] \
                        - points_right[:, 0]
    points_down = np.copy(points_arr)
    points_down[:, 1] = bounding_box[2] \
                      - points_down[:, 1] \
                      - bounding_box[2]
    points_up = np.copy(points_arr)
    points_up[:, 1] = bounding_box[3] \
                    + bounding_box[3] \
                    - points_up[:, 1]
    # Seriously?
    points_arr = np.append(points_arr, np.append(np.append(points_left, points_right, axis=0), np.append(points_down, points_up, axis=0), axis=0), axis=0)
    # Compute Voronoi
    vor = sp.spatial.Voronoi(points_arr)
    # Filter regions
    vertices = []
    centers = []
    for i, center in enumerate(vor.points):
        region_ind = vor.point_region[i]
        # Skip the center if there is no finite region associated with it
        if region_ind == -1:
            continue
        region = vor.regions[region_ind]
        
        # Skip if region is empty
        if not region:
            continue
        
        # Check that the entire region falls in the bounding box
        valid_region = True
        for index in region:
            if index == -1:
                valid_region = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bounding_box[0] - tol <= x and x <= bounding_box[1] + tol and
                       bounding_box[2] - tol <= y and y <= bounding_box[3] + tol):
                    valid_region = False
                    break
        if valid_region:
            vertices.append(vor.vertices[region,:])
            centers.append(center)
    return zip(centers,vertices)




# Right angle on CAB. Distance from C
def right_triangle_dint(pa, pb, pc):
    a = np.linalg.norm(pb-pc)
    b = np.linalg.norm(pa-pc)
    return a*b*np.sqrt(a**2-b**2)/6. \
            + b**3*np.log((a+np.sqrt(a**2-b**2))/b)/6.
    #beta = np.linalg.norm(pa-pb)
    #gamma = np.linalg.norm(pb-pc)
    #return beta**2 * np.sqrt(gamma**2-beta**2) * (
    #return np.sqrt(gamma**2 - beta**2) * (2*beta*gamma - (gamma**2-beta**2)*np.log((gamma-beta)/(gamma+beta))) / 12.0


# Compute the distance integral on general triangles using the
# right-angled triangle function
def triangle_dint(pa, pb, pc, tol=100*eps):
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

# Calculates the expected distance on a polygon relative to pc. poly is an np.array shape=(pts,2)
def poly_dint(poly, pc, tol=100*eps):
    tot = 0.0
    for i in xrange(0, poly.shape[0]):
        tot += triangle_dint(pa=poly[i-1,:], pb=poly[i,:], pc=pc, tol=tol)
    return tot


# Computes the expected minimum distance for a set of points
def expected_dist_conf(points, bounding_box=[0., 1., 0., 1.], draw=False):
    vor = voronoi(points, bounding_box)
    
    if draw:
        ax = plt.gca()
        for (center,vertices) in vor:
            # Plot region triangulation
            for i in xrange(vertices.shape[0]):
                line = np.vstack((vertices[i,:], center))
                ax.plot(line[:,0], line[:,1], color='0.9')
            # Plot region boundary
            aug_vertices = np.append(vertices, [vertices[0]], axis=0)
            ax.plot(aug_vertices[:, 0], aug_vertices[:, 1], color='k',
                    linestyle='-')
            # Plot region center
            ax.plot(center[0], center[1], linestyle='None',
                    marker='o', markersize=6, color='b')
            # Plot region vertices
            #ax.plot(vertices[:, 0], vertices[:, 1], linestyle='None',
            #        marker='o', markersize=4, color='g')
    
    tot = 0.0
    # Calculate total distance integral
    for (center,vertices) in vor:
        tot += poly_dint(vertices, center)
    return tot / ( (bounding_box[1]-bounding_box[0]) \
                  *(bounding_box[3]-bounding_box[2]))





def gen_centers(n=10):
    gen_pts = zip( np.random.uniform(0.0, 1.0, n), np.random.uniform(0.0, 1.0, n) )
    return gen_pts


def test1():
    bounding_box = np.array([0., 1., 0., 1.]) # [x_min, x_max, y_min, y_max]
    servers = gen_centers(n=20)
    
    for _ in xrange(3):
        # Monte-Carlo to check exp_dist
        test_points = gen_centers(n=10000)
        tot = 0.
        for test_point in test_points:
            min_dist = float('inf')
            for server in servers:
                comp_dist = dist.euclidean(server, test_point)
                min_dist = comp_dist if (comp_dist < min_dist) \
                                     else min_dist
            tot += min_dist
        print "MC: ", tot/10000.




    fig, ax = plt.subplots()

    
    st = time.clock()
    exp_dist = expected_dist_conf(servers, bounding_box, draw=True)
    print exp_dist
    print "Time elapsed: ", time.clock()-st

    ax.set_xlim([0.,1.])
    ax.set_ylim([0.,1.])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    plt.savefig("figs/voronoi.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    test1()

