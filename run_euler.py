"""
Copyright © 2021 United States Government as represented 
by the Administrator of the National Aeronautics and Space Administration.
No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
"""

# reconstruct Euler spacetime shock tube data with pretend spacecraft trajectories
# uses euler_numerical_diff.py

import os
import h5py as h5
import numpy as np
import pylab as plt
import tensorflow as tf
#import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from timeit import default_timer as timer
from datetime import timedelta

import euler_numerical_diff as eulermod
import traj_utilities as tj_util

def do_euler_plots(model_sod, path, space,time,st_data,sc_1,sc_2,sc_3,sc_4):
    """
    Make general plots of Sod shocktube and reconstruction
    Plots:
    
    Exact shocktube and trajectories
    Reconstructed shocktube
    Comparison between Exact and Reconstructed at t = 0.16
    Evolution of Reconstructed over time
    
    NOTE: paper plots are found in jupyter notebooks
    """
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sp_min=0
    sp_max=1
    t_min=0
    ax.pcolormesh(space,time,st_data[0].T, cmap=plt.cm.rainbow,vmin=0.,vmax=1.2)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.25)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_xlabel("X")
    ax.set_ylabel("Time")
    ax.set_title('exact density w/ trajectories')
    fig.savefig(path+'/density_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[1].T, cmap=plt.cm.rainbow,vmin=-0.25,vmax=1.25)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.25)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('exact velocity w/ trajectories')
    ax.set_xlabel("X")
    ax.set_ylabel("Time")
    fig.savefig(path+'/v_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[2].T, cmap=plt.cm.rainbow,vmin=0.,vmax=1.2)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.25)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('Exact Pressure w/ trajectories')
    ax.set_xlabel("X")
    ax.set_ylabel("Time")
    fig.savefig(path+'/P_ST_craft.png')
    ax.cla()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    nx_samp = 101
    nt_samp = 101
    sptim_lin = eulermod.generate_spacetime_coloc_linear([[0,1]],[0,0.25],nx_samp,nt_samp).numpy()

    pred_sod = model_sod(sptim_lin).numpy()
    pred_density = pred_sod[:,0].reshape(nx_samp,nt_samp)
    pred_v = pred_sod[:,1].reshape(nx_samp,nt_samp)
    pred_p = pred_sod[:,2].reshape(nx_samp,nt_samp)

    # NN reconstruction of space time
    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_density, vmin=0., vmax=1.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('Predicted Sod Density')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Time")
    fig2.savefig(path+'/density_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_v, vmin=-0.25, vmax=1.25, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('Predicted Sod Velocity')
    fig2.savefig(path+'/v_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_p, vmin=0., vmax=1.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('Predicted Sod Pressure')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Time")
    fig2.savefig(path+'/P_NN_reconstruct.png')
    ax2.cla()

    #exact vs predicted:
    i = np.where(sptim_lin[::nt_samp,1] >= 0.2)[0][0]
    # 200 is index of st_data output where t = 0.2
    ax2.plot(space, st_data[0,:,200], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,0],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1.2)
    ax2.set_title('Density, t=0.2')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Density")
    fig2.savefig(path+'/NN_vs_exact_density.png')
    ax2.cla()

    ax2.plot(space, st_data[1,:,200], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,1],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(-0.25,1.25)
    ax2.set_title('Velocity, t=0.2')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Velocity")
    fig2.savefig(path+'/NN_vs_exact_v.png')
    ax2.cla()

    ax2.plot(space, st_data[2,:,200], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,2],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0.,1.2)
    ax2.set_title('Pressure, t=0.2')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Pressure")
    fig2.savefig(path+'/NN_vs_exact_P.png')
    ax2.cla()

    # array of time evo
    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,0], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,0], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,0], c='k')
    ax2.set_title('Density, t = [0,0.2]')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Density")
    fig2.savefig(path+'/sod_density_intime.png')
    ax2.cla()

    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,1], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,1], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,1], c='k')
    ax2.set_title('Velocity, t = [0,0.2]')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Velocity")
    fig2.savefig(path+'/sod_v_intime.png')
    ax2.cla()

    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,2], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,2], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,2], c='k')
    ax2.set_title('Pressure, t = [0,0.2]')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Pressure")
    fig2.savefig(path+'/sod_p_intime.png')
    ax2.cla()

    #save output prediction to file
    np.save(path+'/sptim_linear.npy', sptim_lin)
    np.save(path+'/pred_sod.npy', pred_sod)

def do_onesc_euler_plots(model_sod, path, space,time,st_data,sc_1):
    """
    Make general plots of Sod shocktube and reconstruction
    (for one spacecraft)
    
    Plots:
    
    Exact shocktube and trajectories
    Reconstructed shocktube
    Comparison between Exact and Reconstructed at t = 0.16
    Evolution of Reconstructed over time
    
    NOTE: paper plots are found in jupyter notebooks
    """
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sp_min=0
    sp_max=1
    t_min=0
    ax.pcolormesh(space,time,st_data[0].T, cmap=plt.cm.rainbow,vmin=0.,vmax=1.2)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.25)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.set_title('Exact Density w/ trajectory')
    ax.set_xlabel("X")
    ax.set_ylabel("Time")
    fig.savefig(path+'/density_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[1].T, cmap=plt.cm.rainbow,vmin=-0.25,vmax=1.25)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.25)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.set_xlabel("X")
    ax.set_ylabel("Time")
    ax.set_title('Exact Velocity w/ trajectory')
    fig.savefig(path+'/v_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[2].T, cmap=plt.cm.rainbow,vmin=0.,vmax=1.2)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.25)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.set_xlabel("X")
    ax.set_ylabel("Time")
    ax.set_title('Exact Pressure w/ trajectory')
    fig.savefig(path+'/P_ST_craft.png')
    ax.cla()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    nx_samp = 101
    nt_samp = 101
    sptim_lin = eulermod.generate_spacetime_coloc_linear([[0,1]],[0,0.25],nx_samp,nt_samp).numpy()

    pred_sod = model_sod(sptim_lin).numpy()
    pred_density = pred_sod[:,0].reshape(nx_samp,nt_samp)
    pred_v = pred_sod[:,1].reshape(nx_samp,nt_samp)
    pred_p = pred_sod[:,2].reshape(nx_samp,nt_samp)

    # NN reconstruction of space time
    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_density, vmin=0., vmax=1.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('Predicted Sod Density')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Time")
    fig2.savefig(path+'/density_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_v, vmin=-0.25, vmax=1.25, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('Predicted Sod Velocity')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Time")
    fig2.savefig(path+'/v_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_p, vmin=0., vmax=1.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('Predicted Sod Pressure')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Time")
    fig2.savefig(path+'/P_NN_reconstruct.png')
    ax2.cla()

    #exact vs predicted:
    i = np.where(sptim_lin[::nt_samp,1] >= 0.2)[0][0]
    # 200 is index of st_data output where t = 0.2
    ax2.plot(space, st_data[0,:,200], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,0],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1.2)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Density")
    ax2.set_title('Density, t=0.2')
    fig2.savefig(path+'/NN_vs_exact_density.png')
    ax2.cla()

    ax2.plot(space, st_data[1,:,200], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,1],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(-0.25,1.25)
    ax2.set_title('Velocity, t=0.2')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Velocity")
    fig2.savefig(path+'/NN_vs_exact_v.png')
    ax2.cla()

    ax2.plot(space, st_data[2,:,200], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,2],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0.,1.2)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Pressure")
    ax2.set_title('Pressure, t=0.2')
    fig2.savefig(path+'/NN_vs_exact_P.png')
    ax2.cla()

    # array of time evo
    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,0], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,0], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,0], c='k')
    ax2.set_title('Density, t = [0,0.2]')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Density")
    fig2.savefig(path+'/sod_density_intime.png')
    ax2.cla()

    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,1], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,1], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,1], c='k')
    ax2.set_title('Velocity, t = [0,0.2]')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Velocity")
    fig2.savefig(path+'/sod_v_intime.png')
    ax2.cla()

    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,2], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,2], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_sod[101*i:(i+1)*101,2], c='k')
    ax2.set_title('Pressure, t = [0,0.2]')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Pressure")
    fig2.savefig(path+'/sod_p_intime.png')
    ax2.cla()

    #save output prediction to file
    np.save(path+'/sptim_linear.npy', sptim_lin)
    np.save(path+'/pred_sod.npy', pred_sod)

    
if __name__ == '__main__':
    # change sc_1 -> sc_n in function load_sp_data
    fname = './sod_spacetime/SOD_SPACETIME.h5'
    space,time,sc_1,sc_2,sc_3,sc_4,st_data,st_input, data_input = tj_util.load_sp_sod_data(fname)
    #space,time,sc_1,st_data,st_input, data_input = tj_util.load_one_sp_sod_data(fname)

    # NN SETUP AND TRAINING
    #mod_name = 'sod_4x16_visc.005_rand1'
    mod_name = 'example'

    model_sod = eulermod.Euler_nd(gamma=5./3.,nh=16, nlayers=4, model_name=mod_name,do_visc=True,visc=0.005, dx=0.001, dt=0.001)
    model_sod.optimizer = tf.optimizers.Adam(lr=1e-3,epsilon=1e-4)

    # define reconstruction domain 0 < x < 1; 0 < t < 0.25
    space_range = tf.convert_to_tensor([[0.002,0.998]], dtype=K.floatx())
    time_range = tf.convert_to_tensor([0.02,0.248],dtype=K.floatx())
    st_input = tf.convert_to_tensor(st_input,dtype=K.floatx())
    data_input = tf.convert_to_tensor(data_input,dtype=K.floatx())

    nc_progression = [30,35,40,50,55,60,65,70,75,80,85]

    # upper limit
    epochs = 300000

    start = timer()
    
    prog = model_sod.train(st_input, data_input, 20, space_range, time_range, epochs=epochs, randomize_ep=1, lr_decay=2./3., anneal_eps=22500, warmup_eps=10000, nc_prog=nc_progression)

    end = timer()
    delta_t = timedelta(seconds=end-start)
    with open(model_sod.log_path+"/timing_file.txt", 'w') as f:
        f.write("epochs: {}; elapsed time: {}\n".format(epochs,delta_t))
    
    model_sod.load_weights(model_sod.log_path+'/best_weights/weights')
    
    path = model_sod.log_path
    do_euler_plots(model_sod,path,space,time,st_data,sc_1,sc_2,sc_3,sc_4)
    #do_onesc_euler_plots(model_sod,path,space,time,st_data,sc_1)
