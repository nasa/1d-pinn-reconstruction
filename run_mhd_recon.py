"""
Copyright © 2021 United States Government as represented 
by the Administrator of the National Aeronautics and Space Administration.
No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
"""

# reconstruct brio-wu spacetime shock tube data with pretend spacecraft trajectories
# uses mhd_numerical_diff.py

import os
import h5py as h5
import numpy as np
import pylab as plt
import tensorflow as tf
#import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from timeit import default_timer as timer
from datetime import timedelta

import mhd_numerical_diff as mhdmod
import traj_utilities as tj_util

def do_rj4a_plots(model_rj4a, path, space,time,st_data,sc_1,sc_2,sc_3,sc_4):
    """
    Make general plots of Ryu-Jones shocktube and reconstruction
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
    ax.set_ylim(t_min, 0.18)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('exact rho w/ trajectories')
    fig.savefig(path+'/rho_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[1].T, cmap=plt.cm.rainbow,vmin=0.,vmax=1.)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.18)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('exact Vx w/ trajectories')
    fig.savefig(path+'/vx_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[2].T, cmap=plt.cm.rainbow,vmin=-1.2,vmax=0.2)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.18)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('exact Vy w/ trajectories')
    fig.savefig(path+'/vy_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[4].T, cmap=plt.cm.rainbow,vmin=0.,vmax=1.2)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.18)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('exact P w/ trajectories')
    fig.savefig(path+'/p_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[6].T, cmap=plt.cm.rainbow,vmin=-0.2,vmax=1.2)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.18)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('exact By w/ trajectories')
    fig.savefig(path+'/by_ST_craft.png')
    ax.cla()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    nx_samp = 100
    nt_samp = 100
    sptim_lin = mhdmod.generate_spacetime_coloc_linear([[0,1]],[0,0.18],nx_samp,nt_samp).numpy()

    pred_rj4a = model_rj4a(sptim_lin).numpy()
    pred_rho = pred_rj4a[:,0].reshape(nx_samp,nt_samp)
    pred_vx = pred_rj4a[:,1].reshape(nx_samp,nt_samp)
    pred_vy = pred_rj4a[:,2].reshape(nx_samp,nt_samp)
    pred_p = pred_rj4a[:,4].reshape(nx_samp,nt_samp)
    pred_by = pred_rj4a[:,6].reshape(nx_samp,nt_samp)
    
    # NN reconstruction of space time
    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_rho, vmin=0., vmax=1.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.18)
    ax2.set_title('predicted rj4a rho')
    fig2.savefig(path+'/rho_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_vx, vmin=0., vmax=1., cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.18)
    ax2.set_title('predicted rj4a vx')
    fig2.savefig(path+'/vx_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_vy, vmin=-1.2, vmax=0.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.18)
    ax2.set_title('predicted rj4a vy')
    fig2.savefig(path+'/vy_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_p, vmin=0., vmax=1.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.18)
    ax2.set_title('predicted rj4a p')
    fig2.savefig(path+'/p_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_by, vmin=-0.2, vmax=1.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.18)
    ax2.set_title('predicted rj4a by')
    fig2.savefig(path+'/by_NN_reconstruct.png')
    ax2.cla()
    
    #exact vs predicted:
    i = np.where(sptim_lin[::nt_samp,1] >= 0.16)[0][0]

    ax2.plot(space, st_data[0,:,81], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,0],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1.2)
    ax2.set_title('rj4a rho, t=0.16')
    fig2.savefig(path+'/NN_vs_exact_rho.png')
    ax2.cla()

    ax2.plot(space, st_data[1,:,81], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,1],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(-0.1,1.1)
    ax2.set_title('rj4a vx, t=0.16')
    fig2.savefig(path+'/NN_vs_exact_vx.png')
    ax2.cla()

    ax2.plot(space, st_data[2,:,81], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,2],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(-1.2,0.2)
    ax2.set_title('rj4a vy, t=0.16')
    fig2.savefig(path+'/NN_vs_exact_vy.png')
    ax2.cla()

    ax2.plot(space, st_data[4,:,81], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,4],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0.,1.2)
    ax2.set_title('rj4a p, t=0.16')
    fig2.savefig(path+'/NN_vs_exact_p.png')
    ax2.cla()

    ax2.plot(space, st_data[6,:,81], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,6],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(-0.2,1.2)
    ax2.set_title('rj4a by, t=0.16')
    fig2.savefig(path+'/NN_vs_exact_by.png')
    ax2.cla()
    
    # array of time evo
    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,0], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,0], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,0], c='k')
    ax2.set_title('rj4a rho, t = [0,0.18]')
    fig2.savefig(path+'/rj4a_rho_intime.png')
    ax2.cla()

    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,1], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,1], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,1], c='k')
    ax2.set_title('rj4a vx, t = [0,0.18]')
    fig2.savefig(path+'/rj4a_vx_intime.png')
    ax2.cla()

    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,2], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,2], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,2], c='k')
    ax2.set_title('rj4a vy, t = [0,0.18]')
    fig2.savefig(path+'/rj4a_vy_intime.png')
    ax2.cla()

    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,4], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,4], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,4], c='k')
    ax2.set_title('rj4a p, t = [0,0.18]')
    fig2.savefig(path+'/rj4a_p_intime.png')
    ax2.cla()

    
    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,6], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,6], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_rj4a[100*i:(i+1)*100,6], c='k')
    ax2.set_title('rj4a by, t = [0,0.18]')
    fig2.savefig(path+'/rj4a_by_intime.png')
    ax2.cla()
    return

def do_brio_plots(model_brio, path, space,time,st_data,sc_1,sc_2,sc_3,sc_4):
    """
    Make general plots of Brio-Wu shocktube and reconstruction
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
    ax.set_title('exact rho w/ trajectories')
    fig.savefig(path+'/rho_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[1].T, cmap=plt.cm.rainbow,vmin=-0.5,vmax=1.)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.25)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('exact Vx w/ trajectories')
    fig.savefig(path+'/vx_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[2].T, cmap=plt.cm.rainbow,vmin=-2.,vmax=0.5)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.25)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('exact Vy w/ trajectories')
    fig.savefig(path+'/vy_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[4].T, cmap=plt.cm.rainbow,vmin=0.,vmax=1.2)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.25)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('exact P w/ trajectories')
    fig.savefig(path+'/p_ST_craft.png')
    ax.cla()

    ax.pcolormesh(space,time,st_data[6].T, cmap=plt.cm.rainbow,vmin=-1.2,vmax=1.2)
    ax.set_xlim(sp_min, sp_max)
    ax.set_ylim(t_min, 0.25)
    ax.plot(sc_1[:,0],sc_1[:,1],color='k')
    ax.plot(sc_2[:,0],sc_2[:,1],color='k')
    ax.plot(sc_3[:,0],sc_3[:,1],color='k')
    ax.plot(sc_4[:,0],sc_4[:,1],color='k')
    ax.set_title('exact By w/ trajectories')
    fig.savefig(path+'/by_ST_craft.png')
    ax.cla()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    nx_samp = 101
    nt_samp = 101
    sptim_lin = mhdmod.generate_spacetime_coloc_linear([[0,1]],[0,0.25],nx_samp,nt_samp).numpy()

    pred_brio = model_brio(sptim_lin).numpy()
    pred_rho = pred_brio[:,0].reshape(nx_samp,nt_samp)
    pred_vx = pred_brio[:,1].reshape(nx_samp,nt_samp)
    pred_vy = pred_brio[:,2].reshape(nx_samp,nt_samp)
    pred_p = pred_brio[:,4].reshape(nx_samp,nt_samp)
    pred_by = pred_brio[:,6].reshape(nx_samp,nt_samp)

    # NN reconstruction of space time
    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_rho, vmin=0., vmax=1.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('predicted brio-wu rho')
    fig2.savefig(path+'/rho_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_vx, vmin=-0.5, vmax=1., cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('predicted brio-wu vx')
    fig2.savefig(path+'/vx_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_vy, vmin=-2., vmax=0.5, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('predicted brio-wu vy')
    fig2.savefig(path+'/vy_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_p, vmin=0., vmax=1.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('predicted brio-wu p')
    fig2.savefig(path+'/p_NN_reconstruct.png')
    ax2.cla()

    ax2.pcolormesh(sptim_lin[:nx_samp,0], sptim_lin[::nt_samp,1], pred_by, vmin=-1.2, vmax=1.2, cmap=plt.cm.rainbow)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.25)
    ax2.set_title('predicted brio-wu by')
    fig2.savefig(path+'/by_NN_reconstruct.png')
    ax2.cla()

    #exact vs predicted:
    i = np.where(sptim_lin[::nt_samp,1] >= 0.2)[0][0]

    ax2.plot(space, st_data[0,:,100], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,0],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1.2)
    ax2.set_title('brio-wu rho, t=0.2')
    fig2.savefig(path+'/NN_vs_exact_rho.png')
    ax2.cla()

    ax2.plot(space, st_data[1,:,100], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,1],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(-0.5,1.)
    ax2.set_title('brio-wu vx, t=0.2')
    fig2.savefig(path+'/NN_vs_exact_vx.png')
    ax2.cla()

    ax2.plot(space, st_data[2,:,100], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,2],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(-2.,0.5)
    ax2.set_title('brio-wu vy, t=0.2')
    fig2.savefig(path+'/NN_vs_exact_vy.png')
    ax2.cla()

    ax2.plot(space, st_data[4,:,100], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,4],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0.,1.2)
    ax2.set_title('brio-wu p, t=0.2')
    fig2.savefig(path+'/NN_vs_exact_p.png')
    ax2.cla()

    ax2.plot(space, st_data[6,:,100], color='k', label='exact')
    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,6],color='r', label='prediction')
    ax2.legend(loc='best')
    ax2.set_xlim(0,1)
    ax2.set_ylim(-1.2,1.2)
    ax2.set_title('brio-wu by, t=0.2')
    fig2.savefig(path+'/NN_vs_exact_by.png')
    ax2.cla()

    # array of time evo
    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,0], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,0], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,0], c='k')
    ax2.set_title('brio-wu rho, t = [0,0.2]')
    fig2.savefig(path+'/brio-wu_rho_intime.png')
    ax2.cla()

    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,1], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,1], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,1], c='k')
    ax2.set_title('brio-wu vx, t = [0,0.2]')
    fig2.savefig(path+'/brio-wu_vx_intime.png')
    ax2.cla()

    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,2], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,2], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,2], c='k')
    ax2.set_title('brio-wu vy, t = [0,0.2]')
    fig2.savefig(path+'/brio-wu_vy_intime.png')
    ax2.cla()

    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,4], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,4], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,4], c='k')
    ax2.set_title('brio-wu p, t = [0,0.2]')
    fig2.savefig(path+'/brio-wu_p_intime.png')
    ax2.cla()


    for i in range(0,nt_samp,8):
        ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,6], ls='-.',lw=1.1)

    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,6], c='k')
    i=0
    ax2.plot(sptim_lin[:nx_samp,0], pred_brio[101*i:(i+1)*101,6], c='k')
    ax2.set_title('brio-wu by, t = [0,0.2]')
    fig2.savefig(path+'/brio-wu_by_intime.png')
    ax2.cla()

    #save output prediction to file
    np.save(path+'/sptim_linear.npy', sptim_lin)
    np.save(path+'/pred_brio.npy', pred_brio)
    

if __name__ == '__main__':
    #problem = 'rj4a'
    problem = 'briowu'
    noise = False
    parallel = True
    
    if problem == 'briowu':
        fname = './brio_wu_spacetime/briowu_ST.h5'
        space,time,sc_1,sc_2,sc_3,sc_4,st_data,st_input, data_input = tj_util.load_sp_mhd_data(fname, 'briowu',parallel=parallel)
        gamma = 2.
    elif problem == 'rj4a':
        fname = './rj4a_spacetime/rj4a_ST.h5'
        space,time,sc_1,sc_2,sc_3,sc_4,st_data,st_input, data_input = tj_util.load_sp_mhd_data(fname, 'rj4a')
        gamma = 5./3.

    if noise:
        data_input = tj_util.add_noise(data_input, std=0.1)
        
    # NN SETUP AND TRAINING
    mod_name = problem+'_example'
    #mod_name = problem+'_scrand_4x48_visc.01'

    model_mhd = mhdmod.MHD_nd(gamma=gamma,nh=48, nlayers=4, model_name=mod_name,do_visc=True,visc=0.01, dx=0.001, dt=0.001)

    model_mhd.optimizer = tf.optimizers.Adam(lr=1e-3,epsilon=1e-4)

    # define reconstruction domain 0 < x < 1; 0 < t < 0.25
    space_range = tf.convert_to_tensor([[0.002,0.998]], dtype=K.floatx())

    if problem == 'briowu':
        time_range = tf.convert_to_tensor([0.02,0.248],dtype=K.floatx())
    elif problem == 'rj4a':
        time_range = tf.convert_to_tensor([0.02,0.18],dtype=K.floatx())
    
    st_input = tf.convert_to_tensor(st_input,dtype=K.floatx())
    data_input = tf.convert_to_tensor(data_input,dtype=K.floatx())

    if problem == 'briowu':
        nc_progression = [30,37,44,51,58,65,72,79,85]
    elif problem == 'rj4a':
        nc_progression = [30,37,44,51,58,65,72]

    # upper limit
    #epochs = 400001
    epochs = 401

    start = timer()

    prog = model_mhd.train(st_input, data_input, 20, space_range, time_range, epochs=epochs, randomize_ep=1, lr_decay=0.6666, anneal_eps=24000, warmup_eps=15000, nc_prog=nc_progression)

    end = timer()
    delta_t = timedelta(seconds=end-start)
    with open(model_mhd.log_path+"/timing_file.txt", 'w') as f:
        f.write("epochs: {}; elapsed time: {}\n".format(epochs,delta_t))
    
    model_mhd.load_weights(model_mhd.log_path+'/best_weights/weights')
    
    path = model_mhd.log_path
    if problem == 'rj4a':
        do_rj4a_plots(model_mhd,path,space,time,st_data,sc_1,sc_2,sc_3,sc_4)
    elif problem == 'briowu':
        do_brio_plots(model_mhd,path,space,time,st_data,sc_1,sc_2,sc_3,sc_4)
