import numpy as np
from numba import jit

#@jit(nopython=True)
def find_neighbors_grid(pos,H,W):
    """
    Computes the list of neighbors of a node in a grid of size HxW.

    Args:
    pos: tuple of two integers, representing the position of the node.
    H: integer, number of rows in the grid.
    W: integer, number of columns in the grid.

    Returns:
    neigh_list: list of tuples of two integers, representing the neighbors of the node.
    """
    neigh_list=[(pos[0]+1,pos[1]),(pos[0]-1,pos[1]),(pos[0],pos[1]+1),(pos[0],pos[1]-1)]
    if pos[0]==0:
        neigh_list.remove((pos[0]-1,pos[1]))
    if pos[0]==H-1:
        neigh_list.remove((pos[0]+1,pos[1]))
    if pos[1]==0:
        neigh_list.remove((pos[0],pos[1]-1))
    if pos[1]==W-1:
        neigh_list.remove((pos[0],pos[1]+1))
    return neigh_list

#@jit(nopython=True)
def eikonal_update(pos, grid, g,h,H,W):
    """
    Updates the value of a node in the grid using the Eikonal equation.
    It computes the solution in U to (Ux-U)^2+(Uy-U)^2=h^2/g^2, where Ux=min(UE,UW) and Uy=min(UN,US) are the neighbouring values used for the update. UN,UE,US,UW are the values of the grid at the north, east, south and west of the node.
    
    Args:
    pos: tuple of two integers, representing the position of the node.
    grid: numpy array of shape (H,W), representing the values of the grid.
    g: float, the velocity (or glide ratio).
    h: float, the spacing of the grid.
    H: integer, number of rows in the grid.
    W: integer, number of columns in the grid.

    Returns:
    new_val: float, the new value of the node.
    """
    y,x=pos
    if y==0:
        US=np.inf
    else:
        US=grid[y-1,x]
    if y==H-1:
        UN=np.inf
    else:  
        UN=grid[y+1,x]
    if x==0:
        UW=np.inf
    else:
        UW=grid[y,x-1]
    if x==W-1:
        UE=np.inf
    else:  
        UE=grid[y,x+1]
    
    Ux=min(UE,UW)
    Uy=min(UN,US)
    if(abs(Ux-Uy)<h/g):
        new_val=(Ux+Uy+np.sqrt(2*(h/g)**2-(Ux-Uy)**2))/2
    else:
        new_val=min(Ux,Uy)+(h/g)

    return new_val

#@jit(nopython=True)
def FMM_MRAP(init_nodes, init_values, elevation, h,g,H,W):
    """
    Variant of the fast marching method (Sethian 1996) to solve the MRAP problem with a uniform velocity field.
    The main difference with the FMM is that when the value of a node is updated, it is updated to the maximum between the value obtained by the Eikonal update and the elevation of the node.

    Args:
    init_nodes: list of tuples of two integers, representing the positions of the sources.
    init_values: list of floats, representing the values of the sources.
    elevation: numpy array of shape (H,W), representing the elevation of the terrain.
    h: float, the spacing of the grid.
    g: float, the velocity (or glide ratio).
    H: integer, number of rows in the grid.
    W: integer, number of columns in the grid.

    Returns:
    grid: numpy array of shape (H,W), representing the minimal altitude at which the glider can be over each node. This is the solution of the MRAP problem.
    """
    H,W=elevation.shape
    grid=np.inf*np.ones((H,W))
    mask_considered=np.zeros((H,W))
    init_accepted=[]
    mask_accepted=np.zeros((H,W))
    for i,node in enumerate(init_nodes):
        mask_accepted[node]=True
        grid[node]=init_values[i]
        init_accepted.append(node)
    
    list_considered=[]
    list_considered_values=[]
    for acc in init_accepted:
        for neigh in find_neighbors_grid(acc,H,W):
            if mask_considered[neigh]==False:
                mask_considered[neigh]=True
                grid_tmp=max(eikonal_update(neigh,grid,g,h,H,W),elevation[neigh])
                if grid_tmp<grid[neigh]:
                    grid[neigh]=grid_tmp
                list_considered.append(neigh)
                list_considered_values.append(grid[neigh])

    
    while list_considered:
        new_acc_idx=np.argmin(np.array(list_considered_values))
        new_acc=list_considered[new_acc_idx]
        mask_accepted[new_acc]=True
        mask_considered[new_acc]=False
        list_considered.remove(new_acc)
        del list_considered_values[new_acc_idx]
        for neigh in find_neighbors_grid(new_acc,H,W):
            if mask_accepted[neigh]==False:
                mask_considered[neigh]=True
                grid_tmp=max(eikonal_update(neigh,grid,g,h,H,W),elevation[neigh])
                if grid_tmp<grid[neigh]:
                    grid[neigh]=grid_tmp
                if neigh not in list_considered:
                    list_considered.append(neigh)
                    list_considered_values.append(grid[neigh])
                else:
                    list_considered_values[list_considered.index(neigh)]=grid[neigh]

    return grid


def point_source_HJB_uniform_solver_MRAP(init_pos,init_altitude,radius,H,W,h,g):
    """
    Solves exactly the HJB equation for MRAP with a point source at pos_seed and a uniform velocity field. Uniform means that the velocity velocity_func(pos,a) is constant in pos, but not necessarily in a.
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

    nx=int(np.ceil(radius/h)) #number of nodes in the x direction at distance at most radius from pos_seed
    ny=int(np.ceil(radius/h))
    for y in range(init_pos[0]-ny,init_pos[0]+ny+1):
        for x in range(init_pos[1]-nx,init_pos[1]+nx+1):
            if((y-init_pos[0])**2+(x-init_pos[1])**2<=(radius/h)**2 and is_in_grid((y,x),H,W) and (y,x)!=init_pos):
                displacement=np.array((h*(y-init_pos[0]),h*(x-init_pos[1])))
                displacement_norm=np.linalg.norm(displacement)
                node_list.append((y,x))
                arrival_times.append(float(init_altitude+displacement_norm/g))
    return node_list,arrival_times

def is_in_grid(pos,H,W):
    return pos[0]>=0 and pos[0]<H and pos[1]>=0 and pos[1]<W

