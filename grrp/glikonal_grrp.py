import numpy as np
from numba import jit

@jit(nopython=True)
def find_neighbors(pos,H,W): 
    """
    Function that returns the list of neighbors of a given position in a triangulated grid graph. 
    The triangulation is obtained by connecting node (i,j) with (i+1,j+1), in addition to the 4 adjacent nodes.

    Args:
    pos: tuple of two integers, representing the position in the graph.
    H: integer, number of rows in the graph.
    W: integer, number of columns in the graph.

    Returns:
    neigh_list: list of tuples of two integers, representing the neighbors of the given position.
    """
    #grid graph with connection also on one of the diagonals
    y,x=pos
    neigh_list=[(y+1,x),(y,x+1),(y-1,x),(y,x-1),(y+1,x+1),(y-1,x-1)]
    #to_remove=[y==H-1,x==W-1,y==0,x==0,x==W-1 or y==H-1, x==0 or y==0]
    if(y==0):
        neigh_list.remove((y-1,x))
        neigh_list.remove((y-1,x-1))

    elif(y==H-1):
        neigh_list.remove((y+1,x))
        neigh_list.remove((y+1,x+1))
        
    if(x==0):
        neigh_list.remove((y,x-1))
        if(y>0):
            neigh_list.remove((y-1,x-1))
        
    elif(x==W-1):
        neigh_list.remove((y,x+1))
        if(y<H-1):
            neigh_list.remove((y+1,x+1))
        
    return  neigh_list
@jit(nopython=True)
def is_in_grid(pos,H,W):
    """
    Function that checks if a given position is inside the grid with dimensions HxW.
    Args:
    pos: tuple of two integers, representing the position in the grid.
    H: integer, number of rows in the grid.
    W: integer, number of columns in the grid.

    Returns:
    boolean: True if the position is inside the grid, False otherwise.
    """
    y,x=pos
    if (y<0 or y>H-1 or x<0 or x>W-1):
        return False
    else:
        return True
@jit(nopython=True)
def is_edge(pos1,pos2):
    """
    Function that checks if two given positions are connected by an edge in the triangulated grid graph.
    The triangulated grid graph is a grid graph with connection also on one of the diagonals.
    In other words triangulation is obtained by connecting node (i,j) with (i+1,j+1), in addition to the 4 adjacent nodes.

    Args:
    pos1: tuple of two integers, representing the first position in the graph.
    pos2: tuple of two integers, representing the second position in the graph.
    Returns:
    boolean: True if the two positions are connected by an edge, False otherwise.
    """
    if(abs(pos1[0]-pos2[0])>1 or abs(pos1[1]-pos2[1])>1 or (pos1[0]-pos2[0])*(pos1[1]-pos2[1])<0 or pos1==pos2 ):
        return False
    else:
        return True
    

@jit(nopython=True)
def compute_near_front(pos, mask_accepted_front, max_dist_edge,hy,hx,H,W):
    """
    Function that computes the list of edges in the accepted front that are close to a given position 'pos'.
    An edge is considered close if at least one of the points on the edge segment is within a distance of max_dist_edge from pos, where h is the diameter of the triangulation.
    Since measuring the distance of pos from edges is hard, we instead neasure the distance of pos from nodes, to decide which edges to include. It is possible however that a point in the edge segment pos1-pos2 is within max_dist_edge from pos, but pos1, pos2 are not (imagine for example if the edge is tangent to the circle of radius max_dist_edge centered at pos). 
    To include also these edges, we include all the nodes that are within a distance of max_dist_edge*sqrt(1+(h/max_dist_edge)^2) from pos. This ensures that at least one of the two endpoints of the edge is included.
    Once the list of nodes within a distance on max_dist_node from pos is computed, we look at all the neighbors of these nodes, to include also the nodes that are not within the circle of radius max_dist_node, but are connected to nodes that are (and therefore the edge segment is partly within the circle of radius max_dist_edge).
    The near front is the portion of accepted front through which the characteristic leading to the point 'pos' can pass. The higher gamma, the more nodes in the near front, as the characteristic can be more angled with respect to the front normal.
    
    Args:
    pos: tuple of two integers, representing the position in the graph for which we want to compute the near front
    mask_accepted_front: numpy array of booleans, representing the accepted front in the graph
    max_dist_edge: float, the maximum euclidean distance from pos for an edge to be considered close
    hx: float, the distance between two adjacent nodes in the x direction
    hy: float, the distance between two adjacent nodes in the y direction
    H: integer, number of rows in the grid.
    W: integer, number of columns in the grid.

    Returns:
    near_front_edges: list of tuples of two tuples of two integers, representing the edges in the accepted front that are in the near front of 'pos'.
    """
    #must find all edges in accepted front for which at least one (also intermediate) point in the edge is within a distance of 2*gamma*h of the considered point. h here is the diameter of the mesh, it's equal to sqrt(hx^2+hy^2).
    #max_dist_edge is 2*gamma*h. The problem is that I measure the distance of nodes from pos, not edges. So I consider nodes within a radius of max_dist_edge*sqrt(1+(h/max_dist_edge)^2)
    triang_diameter=np.sqrt(hx**2+hy**2)
    max_dist_node=max_dist_edge*np.sqrt(1+(triang_diameter/max_dist_edge)**2) #euclidean maximum distance
    max_graph_dist_node_x=int(np.ceil(max_dist_node/hx)) 
    max_graph_dist_node_y=int(np.ceil(max_dist_node/hy)) 
    #print(max_graph_dist_node)
    #enumerate all points in the square of size 2* max_dist_node. keep them if they are part fo the accepted front and if their distance falls within the circle
    near_front_nodes=[]
    #print(max_graph_dist_node)
    for dy in range(-max_graph_dist_node_y,max_graph_dist_node_y+1):
        for dx in range(-max_graph_dist_node_x,max_graph_dist_node_x+1):
            if(is_in_grid((pos[0]+dy,pos[1]+dx),H,W) and mask_accepted_front[pos[0]+dy,pos[1]+dx] and (hy*dy)**2+(hx*dx)**2<=max_dist_node**2):
                near_front_nodes.append((pos[0]+dy,pos[1]+dx))
    #there can still be relevant nodes that we missed, because it can be the case that one node is inside the circle of radius max_dist_node and the other one is not. So we need to look at neighbors of all 
    #print(near_front_nodes)
    new_near_front_nodes=[]
    for pos_nf in near_front_nodes:
        for pos2 in find_neighbors(pos_nf,H,W):
            if(mask_accepted_front[pos2] and pos2 not in near_front_nodes and pos2 not in new_near_front_nodes):
                new_near_front_nodes.append(pos2)
    near_front_nodes.extend(new_near_front_nodes)
    near_front_edges=[]
    for i,pos_nf1 in enumerate(near_front_nodes):
        for pos_nf2 in near_front_nodes[i+1:]: #no edge is counted twice here
            if(is_edge(pos_nf1,pos_nf2)):
                near_front_edges.append((pos_nf1,pos_nf2))
    return near_front_edges


@jit(nopython=True)
def upwind_simplex_update(pos1,pos2,pos,U1,U2,velocity_func,args_velocity_func,hy,hx):
    """
    Function that computes the upwind update of the velocity field at a given position 'pos' in the grid.
    The update is computed by advancing the fornt from the edge pos1-pos2 to the point pos, using the upwind scheme. This consists in finding the optimal point z*pos1+(1-z)*pos2 in the segment pos1-pos2 that minimizes the value of the function V_simplex_zeta(z,pos1,pos2,pos,U1,U2,velocity_func,h).
    The function V_simplex_zeta(z,pos1,pos2,pos,U1,U2,velocity_func,hy,hx) is the arrival time of the characteristic that starts at pos and goes through the point z*pos1+(1-z)*pos2 on the edge pos1-pos2. 
    
    Args:
    pos1: tuple of two integers, representing the first position in the edge.
    pos2: tuple of two integers, representing the second position in the edge.
    pos: tuple of two integers, representing the position in the grid for which we want to compute the upwind update.
    U1: float, value of the arrival time at pos1.
    U2: float, value of the arrival time at pos2.
    velocity_func: function, representing the velocity field of the graph.
    hx: float, the distance between two adjacent nodes in the x direction
    hy: float, the distance between two adjacent nodes in the y direction

    Returns:
    float: the value of the arrival time at the position 'pos' after the upwind update using the edge pos1-pos2.
    """
    np_pos1=np.array(pos1)
    np_pos2=np.array(pos2)
    np_pos=np.array(pos)
    hh=np.array([hy,hx])
    zz=np.linspace(0,1,11) #'11' here is the number of points over which the minimum is taken. The higher the more precise the minimum. We find that increasing this number does not improve the precision of the algorithm, since the error is dominated by other terms.
    V_values=[V_simplex_zeta(z,np_pos1,np_pos2,np_pos,U1,U2,velocity_func,args_velocity_func,hy,hx) for z in zz]
    V_min=min(V_values)
    #the following commented code is an alternative to the previous line. It uses scipy.optimize.minimize_scalar to find the minimum of the function V_simplex_zeta(z,np_pos1,np_pos2,np_pos,U1,U2,velocity_func,h). This is slower than the previous line, but it is more precise.
    #opt_res=optimize.minimize_scalar(lambda z: V_simplex_zeta(z,np_pos1,np_pos2,np_pos,U1,U2,wind_field,hy,hx), bounds=(0,1),method='bounded',options={'maxiter':5})
    #V_min=opt_res.fun
    z_min=zz[V_values.index(V_min)]
    pos_zmin=z_min*np_pos1+(1-z_min)*np_pos2
    char_vec=(np_pos-pos_zmin)*hh
    return V_min,char_vec/np.linalg.norm(char_vec)
    


@jit(nopython=True)
def V_simplex_zeta(z,pos1,pos2,pos,U1,U2,velocity_func,args_velocity_func,hy,hx):
    """
    Function that computes the arrival time of the characteristic that starts at pos and goes through the point z*pos1+(1-z)*pos2 on the edge segment pos1-pos2.
    
    Args:
    z: float, parameter in the segment pos1-pos2 that defines the point z*pos1+(1-z)*pos2.
    pos1: numpy array of two integers, representing the first position in the edge.
    pos2: numpy array of two integers, representing the second position in the edge.
    pos: numpy array of two integers, representing the position in the grid for which we want to compute the arrival time.
    U1: float, value of the arrival time at pos1.
    U2: float, value of the arrival time at pos2.
    velocity_func: function, representing the velocity field (dependent on the position and on the direction) of the graph.
    hx: float, the distance between two adjacent nodes in the x direction
    hy: float, the distance between two adjacent nodes in the y direction

    Returns:
    float: the value of the arrival time of the characteristic that starts at pos and goes through the point z*pos1+(1-z)*pos2.
    """
    scale=np.array([hy,hx])
    a_unnorm=scale*(pos-z*pos1-(1-z)*pos2)
    tau=np.linalg.norm(a_unnorm)
    return tau/velocity_func(a_unnorm/tau,args_velocity_func)+z*U1+(1-z)*U2  


def OUM_GRRP(velocity_func,Gamma,elevation,init_nodes,init_altitudes,init_chars,hy,hx,hz,H,W,wind_field):
    """
    Function that computes the solution of the Hamilton-Jacobi-Bellman with Hamiltonian min_a((grad_x U(x))*a f(x,a))=1 equation on a triangulated grid graph. a is a vector in the unit circle, f(x,a) is the velocity field (encoded in velocity_func), dependent on the position and on the direction, '*' represents the 2d dot product. We solve for U(x), the arrival time at the node x.
    The solution is computed using the upwind scheme, with a relaxed variant that uses the near front of each node to compute the value of the node. The relaxation consists in computing the value of the arrival time of a node only once.
    This function implements a solver for GRRP based on the algorithm M3 in the paper "ORDERED UPWIND METHODS FOR STATIC HAMILTON–JACOBI EQUATIONS: THEORY AND ALGORITHMS" by Sethian and Vladimirsky 2001.
    As h goes to zero (and H,W go to infinity), this algorithm converges to the viscosity solution of the HJB equation.
    
    Args:   
    velocity_func: function, representing the velocity field of the graph as a function of the position and the direction. takes as input a tuple of two integers and a numpy array of two floats, and returns a positive float.
    Gamma: float, the parameter that controls the anisotropy of the problem. Gamma= max(velocity_func)/min(velocity_func) where the max and min are taken over the direction. The higher Gamma, the more nodes in the near front of a node are considered.
    elevation: numpy array of shape (H,W),representing the elevation of the terrain at each node in the grid.
    init_nodes: list of tuples of two integers, representing the initial conditions (i.e., nodes for which the arrival time is known).
    init_values: list of floats, of same length as init_nodes. Represents the values of the arrival time at each of the init_nodes.
    hx: float, the distance between two adjacent nodes in the x direction
    hy: float, the distance between two adjacent nodes in the y direction
    hz: vertical resolution of the wind field. 
    H: integer, number of rows in the graph.
    W: integer, number of columns in the graph.

    Returns:
    grid: numpy array of floats, representing the values of the arrival time at each node in the graph.
    char_grid: numpy array of size (H,W,2). char_grid[y,x] is a 2d vector of unit norm representing the direction of the characteristic passing through position y,x.
    """

    elevation=-elevation
    triang_diameter=np.sqrt(hx**2+hy**2) #diameter of the triangulation
    max_dist_edge=2*Gamma*triang_diameter #this parameter controls how big the near front of a considered node is. 
    mask_accepted=np.full((H,W),False)
    mask_accepted_front=np.full((H,W),False)
    mask_considered=np.full((H,W),False)
    considered=[]
    considered_values=[]
    grid=np.inf*np.ones((H,W))
    char_grid=np.zeros((H,W,2))
    #initialize accepted nodes
    for i,pos in enumerate(init_nodes):
        y,x=pos
        grid[y,x]=-init_altitudes[i]
        char_grid[y,x]=init_chars[i]
        mask_accepted[y,x]=True
        
    #initialize considered nodes
    for pos_acc in init_nodes:
        for pos in find_neighbors(pos_acc,H,W):
            if not mask_accepted[pos]:
                mask_accepted_front[pos_acc]=True
                if not mask_considered[pos]:
                    considered.append(pos)
                    considered_values.append(np.inf)
                    mask_considered[pos]=True
    for i, pos in enumerate(considered):
        near_front_edges=compute_near_front(pos, mask_accepted_front, max_dist_edge,hy,hx,H,W)
        #proxy_altitude=min([grid[pos1] for pos1,pos2 in near_front_edges]) #proxy altitude is a proxy for grid[pos]
        proxy_altitude=min(min([grid[pos1] for pos1,pos2 in near_front_edges]),min([grid[pos2] for pos1,pos2 in near_front_edges]))#proxy altitude is a proxy for grid[pos]

        wind=wind_field[pos[0],pos[1],np.round(-proxy_altitude/hz).astype(int)] 
        possible_altitudes,possible_chars=tuple(zip(*(upwind_simplex_update(pos1,pos2,pos,U1=grid[pos1],U2=grid[pos2],velocity_func=velocity_func,args_velocity_func=wind,hy=hy,hx=hx) for pos1,pos2 in near_front_edges)))
        grid_tmp=min(possible_altitudes)
        char_tmp=possible_chars[possible_altitudes.index(grid_tmp)]
        #grid_tmp,char_tmp=min((upwind_simplex_update(pos1,pos2,pos,U1=grid[pos1],U2=grid[pos2],velocity_func=velocity_func,args_velocity_func=wind,hy=hy,hx=hx) for pos1,pos2 in near_front_edges))
        if(grid_tmp<=elevation[pos]):
            grid[pos]=grid_tmp
            char_grid[pos]=char_tmp
            considered_values[i]=grid[pos]
        else:
            raise Exception("ERROR: initialization too close to obstacle") #this error gets raised if one of the neighbours of the init_nodes cannot be reached by the aircraft. To avoid this error use a finer mesh.
    #finished initializing the values
  
    while considered: 
        #start_time_iter=time.time()
        index_min_considered=np.argmin(np.array(considered_values))
        #index_min_considered = min(range(len(considered_values)), key=considered_values.__getitem__) #find the index of the minimum value in the list. this operation could be optimized by using a heap
        
        _=considered_values.pop(index_min_considered) #remove the minimum value from the list
        new_acc_pos=considered.pop(index_min_considered) #remove the corresponding position from the list
        mask_considered[new_acc_pos]=False #remove the position from the considered set
        mask_accepted[new_acc_pos]=True #add the position to the accepted set
        flag_accepted_front=False
        #update the accepted front
        for pos in find_neighbors(new_acc_pos,H,W): #check if the neighbours of pos that were in accepted_front still are (i.e. check if they still have one considered neighbour)
            #eliminate the nodes that are no longer in the accepted front because they don't have any considered neighbors anymore
            if mask_accepted_front[pos]:
                remove_pos_from_front=True
                for neigh in find_neighbors(pos,H,W):
                    if mask_considered[neigh]:
                        remove_pos_from_front=False
                        break
                if remove_pos_from_front:
                    mask_accepted_front[pos]=False

        mask_accepted_front[new_acc_pos]=True #temptatively add the new accepted node to the accepted front. A node should be in the accepted front exclusively if one of its neighbours is considered. We now loop over all neighbours of new_acc_pos to check if they can be added to the considered nodes. If none of them can be added to the considered nodes, we remove new_acc_pos from the accepted front.
        for pos in find_neighbors(new_acc_pos,H,W):
            if mask_considered[pos]: #if new_acc_pos has at least one considered neighbor, add it to the accepted front
                flag_accepted_front=True

            if not mask_considered[pos] and not mask_accepted[pos]: #if the node is not already in the considered set (and it's not accepted), compute its value and if this is above the elevation, add it to the considered set
                    #start_time_update=time.time()
                    near_front_edges=compute_near_front(pos, mask_accepted_front, max_dist_edge,hy,hx,H,W)
                    if near_front_edges:
                        proxy_altitude=min(min([grid[pos1] for pos1,pos2 in near_front_edges]),min([grid[pos2] for pos1,pos2 in near_front_edges]) )#proxy altitude is a proxy for grid[pos]
                        wind=wind_field[pos[0],pos[1],np.round(-proxy_altitude/hz).astype(int)]
                        possible_altitudes,possible_chars=tuple(zip(*(upwind_simplex_update(pos1,pos2,pos,U1=grid[pos1],U2=grid[pos2],velocity_func=velocity_func,args_velocity_func=wind,hy=hy,hx=hx) for pos1,pos2 in near_front_edges)))
                        grid_tmp=min(possible_altitudes)
                        char_tmp=possible_chars[possible_altitudes.index(grid_tmp)]
                        if(grid_tmp<elevation[pos]):
                            considered.append(pos)
                            mask_considered[pos]=True
                            grid[pos]=grid_tmp
                            char_grid[pos]=char_tmp
                            considered_values.append(grid[pos])
                            flag_accepted_front=True
        if not flag_accepted_front: #if none of the neighbors of new_acc_pos can be added to the considered set, remove new_acc_pos from the accepted front, since none of its neighbors are considered
            mask_accepted_front[new_acc_pos]=False
        
                    #time_oum_update.append(time.time()-start_time_update-time_near_front[-1])
                    
            #the non relaxed variant here would require to update the value of all considered nodes that are have new_acc_pos in their near front. Instead of doing this we just use a wider near front for each considered node, and compute its value only once.
        #iteration_times.append(time.time()-start_time_iter)
    return -grid,char_grid#, time_init, time_search_min_considered, time_near_front, time_oum_update, iteration_times

def point_source_HJB_uniform_solver(init_pos,init_altitude,radius,H,W,hy,hx,velocity_func):
    """
    Solves exactly the HJB equation for GRRP with a point source at pos_seed and a uniform velocity field. Uniform means that the velocity velocity_func(pos,a) is constant in pos, but not necessarily in a.
    The solution is computed on a grid of size HxW nodes, with x and y spacing respectively hx and hy.
    In this case the solution is easy because the characteristics are just straight lines, so the solution at x is just the distance between x and the source, divided by the velocity in direction x-source.
    
    Args:
    pos_seed: tuple of two integers, representing the position of the source.
    radius: float, the radius  around pos_seed where the solution is computed.
    H: integer, number of rows in the grid.
    W: integer, number of columns in the grid.
    hx: float, the x-spacing of the grid.
    hy: float, the y-spacing of the grid.
    velocity_func: function, representing the velocity field of the graph as a function of the position and the direction. It is assumed that the velocity does not depend on the position. Takes as input a tuple of two integers (the position x) and a numpy array of two floats (the direction a), and returns a positive float.
   
    Returns:
    node_list: list of tuples of two integers, representing the nodes in the grid at distance at most radius from pos_seed.
    values: list of floats, representing the value of the arrival time at each node in node_list.
    """
    node_list=[init_pos]
    arrival_times=[init_altitude]
    char_list=[np.array([0,0])]

    nx=int(np.ceil(radius/hx)) #number of nodes in the x direction at distance at most radius from pos_seed
    ny=int(np.ceil(radius/hy))
    for y in range(init_pos[0]-ny,init_pos[0]+ny+1):
        for x in range(init_pos[1]-nx,init_pos[1]+nx+1):
            if(hy**2*(y-init_pos[0])**2+hx**2*(x-init_pos[1])**2<=radius**2 and is_in_grid((y,x),H,W) and (y,x)!=init_pos):
                displacement=np.array((hy*(y-init_pos[0]),hx*(x-init_pos[1])))
                displacement_norm=np.linalg.norm(displacement)
                node_list.append((y,x))
                char_list.append(displacement/displacement_norm) 
                arrival_times.append(init_altitude-displacement_norm/velocity_func(displacement/displacement_norm))
    return node_list,arrival_times,char_list 
