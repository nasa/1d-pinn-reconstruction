"""
Copyright © 2021 United States Government as represented 
by the Administrator of the National Aeronautics and Space Administration.
No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
"""

"""
common functions for sampling spacetime data with "spacecraft" trajectories
"""

import h5py as h5
import numpy as np
import scipy.ndimage as ndi

def find_pixel_bounds(arr, loc):
    if isinstance(loc, float) or isinstance(loc, int):
        return find_pixel_bounds_single(arr, loc)
    else:
        return find_pixel_bounds_arr(arr, loc)

def find_pixel_bounds_arr(arr, locs):
    idxs = []
    for loc in locs:
        idxs.append(find_pixel_bounds_single(arr,loc))

    return idxs

def find_pixel_bounds_single(arr, loc):
    """
    arr: sorted array of values
    loc: single value
    Given loc and arr, return fractional index of loc in arr
    E.g. arr = [1,2,3]; loc = 1.5 -> idx = 0.5
    If loc is outside bounds of arr, return -1 if loc < arr[0] or len(arr) if loc > arr[-1]
    """
    try:
        idx_r = np.where(arr >= loc)[0][0]
    except IndexError:
        idx_r = len(arr)

    try:
        idx_l = np.where(arr <= loc)[0][-1]
    except IndexError:
        idx_l = -1

    if idx_l == -1:
        return idx_l
    elif idx_r == len(arr):
        return idx_r
    elif idx_l == idx_r:
        return idx_l
    else:
        return idx_l + (loc-arr[idx_l])/(arr[idx_r]-arr[idx_l])

def transform_to_pixel_coords(space, time, x, t):
    """
    transforms input (x,t) into pixel coordinates (fractional indexes)
    based on input space/time arrays
    """
    idx_x = find_pixel_bounds(space,x)
    idx_t = find_pixel_bounds(time,t)

    return idx_x, idx_t

def data_along_trajectory(space, time, spacetime_data, xt,tt, order=3):
    """
    Interpolates given data to given trajectory
    
    space,time : input arrays of space/time coordinates
    spacetime_data: input data corresponding to space/time coordinates
    xt, tt: trajectory space/time coordinates
    
    returns data corresponding to trajectory space/time
    """
    idx_x, idx_t = transform_to_pixel_coords(space, time, xt,tt)

    return ndi.map_coordinates(spacetime_data, [idx_x, idx_t], order=order)

def linear_data(x,y,npts):
    """
    Returns column stack of linear (x,y) data; line is from (x1,y1) to (x2,y2)
    x: (x1,x2)
    y: (y1,y2)
    npts: number of points in returned data
    """
    x1,x2 = x
    y1, y2 = y
    slope = (y2-y1)/(x2-x1) if x2-x1 != 0. else 0.
    x_sp = np.linspace(x1,x2,npts)

    y_sp = y1+slope*(x_sp-x1)

    return np.column_stack((x_sp,y_sp))

def spacecraft_sod_data(space, time, st_data, traj):
    """
    returns data sampled from `st_data` along trajectory `traj`.
    idx 0: density
    idx 1: velocity
    idx 2: pressure
    """
    dat_0 = data_along_trajectory(space, time, st_data[0], traj[:,0], traj[:,1])
    dat_1 = data_along_trajectory(space, time, st_data[1], traj[:,0], traj[:,1])
    dat_2 = data_along_trajectory(space, time, st_data[2], traj[:,0], traj[:,1])

    return np.column_stack((dat_0, dat_1, dat_2))

def load_sp_sod_data(fle):
    """
    Given filename `fle` corresponding to Sod shocktube spacetime data,
    return:
    
    space: 1D spatial coordinates of shocktube
    time: 1D temporal coordinates of shocktube
    sc_1 -> sc_4 : four spacecraft trajectories (x,t) through spacetime
    st_data: raw plasma variables from shocktube
    st_input: stacked spacecraft space/time data for input into NN
    data_input: stacked plasma data associated with st_input for input into NN
    
    NOTE: manually change definitions of sc_1 -> sc_4 to adjust trajectories
    """
    # spacetime data
    euler_st = h5.File(fle, 'r')

    space = euler_st['x'][:]
    time = euler_st['time'][:]
    st_data = np.array(euler_st['U'])
    euler_st.close()

    nx = space.shape[0]
    nt = time.shape[0]

    sp_min = space[0]
    sp_max = space[-1]

    t_min = time[0]
    t_max = time[-1]

    # sample from st data

    # #semi random
    # sc_1 = linear_data((0.002,0.998),(0.05,0.2),150)
    # sc_2 = linear_data((0.002,0.75),(0.002,0.248),150)
    # sc_3 = linear_data((0.65,0.35),(0.002,0.248),150)
    # sc_4 = linear_data((0.998,0.0022),(0.12,0.248),150)

    #parallel over (0,1)X(0,0.25) (x,t) domain
    sc_1 = linear_data((0.002,0.478),(0.175,0.248),75)
    sc_2 = linear_data((0.002,0.998),(0.045,0.2),75)
    sc_3 = linear_data((0.002,0.902),(0.11,0.248),75)
    sc_4 = linear_data((0.1306,0.998),(0.002,0.135),75)

    #interpolate from data
    U_sc1 = spacecraft_sod_data(space, time, st_data, sc_1)
    U_sc2 = spacecraft_sod_data(space, time, st_data, sc_2)
    U_sc3 = spacecraft_sod_data(space, time, st_data, sc_3)
    U_sc4 = spacecraft_sod_data(space, time, st_data, sc_4)

    #stack and return
    st_input = np.concatenate((sc_1, sc_2, sc_3, sc_4))
    data_input = np.concatenate((U_sc1, U_sc2, U_sc3, U_sc4))

    return space,time,sc_1,sc_2,sc_3,sc_4,st_data,st_input, data_input

def load_one_sp_sod_data(fle):
    """
    Given filename `fle` corresponding to Sod shocktube spacetime data,
    return:
    
    space: 1D spatial coordinates of shocktube
    time: 1D temporal coordinates of shocktube
    sc_1: one spacecraft trajectory (x,t) through spacetime
    st_data: raw plasma variables from shocktube
    st_input: spacecraft space/time data for input into NN
    data_input: plasma data associated with st_input for input into NN
    
    NOTE: manually adjust sc_1 to change trajectory
    
    """
    # spacetime data
    euler_st = h5.File(fle, 'r')

    space = euler_st['x'][:]
    time = euler_st['time'][:]
    st_data = np.array(euler_st['U'])
    euler_st.close()

    nx = space.shape[0]
    nt = time.shape[0]

    sp_min = space[0]
    sp_max = space[-1]

    t_min = time[0]
    t_max = time[-1]

    # sample from st data

    #single diagnoal over (0,1)X(0,0.25) (x,t) domain
    sc_1 = linear_data((0.002,0.998),(0.02,0.248),75)

    #interpolate from data
    U_sc1 = spacecraft_sod_data(space, time, st_data, sc_1)

    #stack and return
    st_input = sc_1
    data_input = U_sc1

    return space,time,sc_1,st_data,st_input, data_input

def load_two_sp_sod_data(fle):
    """
    Given filename `fle` corresponding to Sod shocktube spacetime data,
    return:
    
    space: 1D spatial coordinates of shocktube
    time: 1D temporal coordinates of shocktube
    sc_1 -> sc_2 : two spacecraft trajectories (x,t) through spacetime
    st_data: raw plasma variables from shocktube
    st_input: stacked spacecraft space/time data for input into NN
    data_input: stacked plasma data associated with st_input for input into NN
    
    NOTE: manually adjust sc_1/sc_2 to change trajectories
    """
    # spacetime data
    euler_st = h5.File(fle, 'r')

    space = euler_st['x'][:]
    time = euler_st['time'][:]
    st_data = np.array(euler_st['U'])
    euler_st.close()

    nx = space.shape[0]
    nt = time.shape[0]

    sp_min = space[0]
    sp_max = space[-1]

    t_min = time[0]
    t_max = time[-1]

    # sample from st data

    #single diagnoal over (0,1)X(0,0.25) (x,t) domain
    sc_1 = linear_data((0.002,0.998),(0.02,0.248),75)
    sc_2 = linear_data((0.4,0.998),(0.002,0.14),75)
    
    #interpolate from data
    U_sc1 = spacecraft_sod_data(space, time, st_data, sc_1)
    U_sc2 = spacecraft_sod_data(space, time, st_data, sc_2)

    #stack and return
    st_input = np.concatenate((sc_1, sc_2))
    data_input = np.concatenate((U_sc1, U_sc2))

    return space,time,sc_1,sc_2,st_data,st_input, data_input


def spacecraft_mhd_data(space, time, st_data, traj):
    """
    returns data sampled from `st_data` along trajectory `traj`.
    idx 0: density
    idx 1: v_x
    idx 2: v_y
    idx 3: v_z
    idx 4: pressure
    idx 5: b_x
    idx 6: b_y
    idx 7: b_z
    """
    dat_0 = data_along_trajectory(space, time, st_data[0], traj[:,0], traj[:,1])
    dat_1 = data_along_trajectory(space, time, st_data[1], traj[:,0], traj[:,1])
    dat_2 = data_along_trajectory(space, time, st_data[2], traj[:,0], traj[:,1])
    dat_3 = data_along_trajectory(space, time, st_data[3], traj[:,0], traj[:,1])
    dat_4 = data_along_trajectory(space, time, st_data[4], traj[:,0], traj[:,1])
    dat_5 = data_along_trajectory(space, time, st_data[5], traj[:,0], traj[:,1])
    dat_6 = data_along_trajectory(space, time, st_data[6], traj[:,0], traj[:,1])
    dat_7 = data_along_trajectory(space, time, st_data[7], traj[:,0], traj[:,1])

    return np.column_stack((dat_0, dat_1, dat_2, dat_3, dat_4, dat_5, dat_6, dat_7))

def load_sp_mhd_data(fle, problem, parallel=False):
    """
    Given filename `fle` corresponding to shocktube spacetime data,
    `problem` = 'briowu' or 'rj4a'
    return:
    
    space: 1D spatial coordinates of shocktube
    time: 1D temporal coordinates of shocktube
    sc_1 -> sc_4 : four spacecraft trajectories (x,t) through spacetime
    st_data: raw plasma variables from shocktube
    st_input: stacked spacecraft space/time data for input into NN
    data_input: stacked plasma data associated with st_input for input into NN
    
    NOTE: manually adjust sc_1 - sc_4 to change trajectories
    `parallel` keyword is used for my convenience
    """
    # spacetime data
    mhd_st = h5.File(fle, 'r')

    space = mhd_st['x'][:]
    time = mhd_st['t'][:]
    st_data = np.array(mhd_st['U'])
    mhd_st.close()

    nx = space.shape[0]
    nt = time.shape[0]

    sp_min = space[0]
    sp_max = space[-1]

    t_min = time[0]
    t_max = time[-1]

    # sample from st data

    if(problem == 'briowu'):
        if parallel:
            #parallel over (0,1)X(0,0.25) (x,t) domain
            sc_1 = linear_data((0.002,0.478),(0.175,0.248),75)
            sc_2 = linear_data((0.002,0.998),(0.045,0.2),75)
            sc_3 = linear_data((0.002,0.902),(0.11,0.248),75)
            sc_4 = linear_data((0.1306,0.998),(0.002,0.135),75)
        else:
            #semi random
            sc_1 = linear_data((0.002,0.998),(0.05,0.2),150)
            sc_2 = linear_data((0.002,0.75),(0.002,0.248),150)
            sc_3 = linear_data((0.65,0.35),(0.002,0.248),150)
            sc_4 = linear_data((0.998,0.0022),(0.12,0.248),150)
            
    elif(problem == 'rj4a'):
        # #semi random (max time = 0.18 for rj 4a)
        sc_1 = linear_data((0.002,0.998),(0.05,0.12),75)
        sc_2 = linear_data((0.002,0.75),(0.002,0.178),75)
        sc_3 = linear_data((0.65,0.35),(0.002,0.178),75)
        sc_4 = linear_data((0.998,0.0022),(0.12,0.178),75)

    #interpolate from data
    U_sc1 = spacecraft_mhd_data(space, time, st_data, sc_1)
    U_sc2 = spacecraft_mhd_data(space, time, st_data, sc_2)
    U_sc3 = spacecraft_mhd_data(space, time, st_data, sc_3)
    U_sc4 = spacecraft_mhd_data(space, time, st_data, sc_4)

    #stack and convert to tf tensors
    st_input = np.concatenate((sc_1, sc_2, sc_3, sc_4))
    data_input = np.concatenate((U_sc1, U_sc2, U_sc3, U_sc4))

    return space,time,sc_1,sc_2,sc_3,sc_4,st_data,st_input, data_input

def add_noise(data, std=0.03):
    """
    Given `data`, add random Gaussian noise with standard deviation `std`
    """
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, std, size=data.shape) #roughly -3*std < noise < 3*std

    # enforce positive density/pressure (set to zero)
    noisy_data = data+noise
    noisy_data[:,0] = np.clip(noisy_data[:,0], a_min=0., a_max=None)
    noisy_data[:,4] = np.clip(noisy_data[:,4], a_min=0., a_max=None)
    
    return noisy_data
