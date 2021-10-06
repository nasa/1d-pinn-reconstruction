"""
Copyright © 2021 United States Government as represented 
by the Administrator of the National Aeronautics and Space Administration.
No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
"""

import numpy as np
import tensorflow as tf
import datetime
import os
import pickle
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
#import tensorflow_addons as tfa
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
tf.keras.backend.set_floatx('float32')
from timeit import default_timer as timer
from datetime import timedelta

def generate_spacetime_coloc_linear(space, time, n_coll, n_tcoll):
    """
    Returns tensorflow tensor (ncoll*nt_coll, ndim+1) of spacetime
    linearly distributed between [space0, space1] and [time0, time1]
    """
    ndim = len(space)

    space_coll = np.expand_dims(np.linspace(space[0][0],space[0][1],n_coll),1)

    if(ndim > 1):
        space_coll = np.column_stack((space_coll, np.linspace(space[1][0],space[1][1],n_coll)))


    if(ndim > 2):
        space_coll = np.column_stack((space_coll, np.linspace(space[2][0],space[2][1],n_coll)))


    time_coll = np.linspace(time[0],time[1],n_tcoll)

    spacetime_coll = np.tile(space_coll, reps=[n_tcoll,1])
    spacetime_coll = np.column_stack((spacetime_coll, np.zeros((spacetime_coll.shape[0],1))))

    for i in range(n_tcoll):
        spacetime_coll[(i)*n_coll:(i+1)*n_coll,-1] = time_coll[i]

    return tf.convert_to_tensor(spacetime_coll, dtype=K.floatx())

def generate_spacetime_coloc_rand(space, time, n_coll_dim, tensor=False):
    """
    Given N-dimensional array of space begin and end coordinates (e.g. [x0,x1])
    and 1D array of time begin and end ([t0,t1])
    return tensorflow tensor of n_coll_dim**2 randomly sampled points (x,t)
    with shape (n_coll_dim**N, N+1) 
    """
    ndim = len(space)
    coloc_pts = lhsamp(ndim+1,n_coll_dim**(ndim+1)) #ndim + 1 time dim

    for i in range(ndim):
        coloc_pts[:,i] = coloc_pts[:,i]*(space[i][1] - space[i][0]) + space[i][0]

    coloc_pts[:,-1] = coloc_pts[:,-1]*(time[1]-time[0])+time[0]
    if tensor:
        coloc_pts = tf.convert_to_tensor(coloc_pts, dtype=K.floatx())

    return coloc_pts

#taken from pydoe2._lhsclassic
# Original Code: Abraham Lee, Michael Baudin, Maria Christopoulou, Yann Collette
# and Jean-Marc Martinez
# https://github.com/clicumu/pyDOE2/blob/master/pyDOE2/doe_lhs.py
# under BSD-3 license
def lhsamp(n, samples, randomstate=np.random.RandomState()):
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    u = randomstate.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a

    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = randomstate.permutation(range(samples))
        H[:, j] = rdpoints[order, j]

    return H

# trainable tanh activation function
# uses node-individual beta
def modtanh(x,b):
    return tf.math.tanh(b*x)

#uses individual betas, one for each node
class ModTanh_all(Layer):
    def __init__(self, output_dim, beta=1.,trainable=False, **kwargs):
        super(ModTanh_all, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable
        self.output_dim = output_dim

    def build(self, input_shape):
        self.beta_fac = self.add_weight("beta", shape=[1, self.output_dim], trainable=self.trainable, dtype=K.floatx(), initializer=tf.constant_initializer(self.beta))
        #if self.trainable:
            #self._trainable_weights.append(self.beta_fac)

        super(ModTanh_all, self).build(input_shape)
        self.built=True

    def call(self, inputs, mask=None):
        return modtanh(inputs, self.beta_fac)

    def compute_output_shape(self,input_shape):
        return (input_shape[-1], self.output_dim)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(ModTanh_all, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class Euler_nd(Model):

    def __init__(self, gamma, input_shape_dim=(2,), output_dim=3, nh = 32, nlayers=4, path=None, model_name=None, visc=0., do_visc=False, loss_type='MSE', dx = 0.001, dt = 0.001):
        """
        NOTE: This will save model information and checkpoints in './euler_models/eulernd1/'+model_name_+'/' by default.
        Default model name is 'Euler_recon_'+str(nlayers)+'x'+str(nh)'
        NN architecture defined as in paper
        "Neural Network Reconstruction of Plasma Space-Time" by C.Bard and J. Dorelli (DOI: 10.3389/fspas.2021.732275)
        
        gamma : ratio of specific heats
        input_shape_dim : tuple denoting number of dimensions of input spacetime
        output_dim : number of output variables in plasma state vector U
        nh = number of nodes per layer
        nlayers : number of layers
        lambda_phys : regularization parameter between data loss and phys loss
                      (applied to phys loss)
        path : (optional) folder to save model/training information/output
        model_name : (optional) name of model; saved in path
        visc : setting for viscosity term nu
        do_visc : True/False
        loss_type : 'MSE' or 'logcosh'
        dx : resolution in space
        dt : resolution in time
        """
        
        super(Euler_nd, self).__init__()
        self.min_loss = 10000.
        self.ckpt_min_loss = 10000.
        self.do_visc=do_visc
        self.gamma = gamma
        self.input_shape_dim = input_shape_dim
        self.output_dim = output_dim
        self.tosave = True
        self.visc = tf.constant(visc, dtype=K.floatx())
        self.dx_arr = tf.constant((dx,0), dtype=K.floatx())
        self.dt_arr = tf.constant((0,dt), dtype=K.floatx())
        self.dx = tf.constant(dx, dtype=K.floatx())
        self.dt = tf.constant(dt, dtype=K.floatx())

        self.optimizer = None

        if loss_type.lower() == 'logcosh':
            self.loss_type = self.loss_logcosh
        else:
            self.loss_type = self.loss_MSE

        # interconnected network (Wang et al. 2020)
        self.U = Dense(nh, input_shape=input_shape_dim, dtype=K.floatx())
        self.U_act = ModTanh_all(nh, beta=1., trainable=True, dtype=K.floatx())

        self.V = Dense(nh, input_shape=input_shape_dim, dtype=K.floatx())
        self.V_act = ModTanh_all(nh, beta=1., trainable=True, dtype=K.floatx())

        self.base_layers = []
        self.act_layers = []
        self.base_layers.append(Dense(nh, input_shape=input_shape_dim, dtype=K.floatx()))
        self.act_layers.append(ModTanh_all(nh, beta=1., trainable=True, dtype=K.floatx()))

        for i in range(nlayers-1):
            self.base_layers.append(Dense(nh, input_shape=(nh,), dtype=K.floatx()))
            self.act_layers.append(ModTanh_all(nh,beta=1., trainable=True, dtype=K.floatx()))

        self.nlayers =nlayers

        #output layer
        self.output_layer = Dense(output_dim, input_shape=(nh,), activation = 'linear', dtype=K.floatx())

        # metrics
        if model_name is None:
            model_name_ = 'Euler_recon_'+str(nlayers)+'x'+str(nh)
        else:
            model_name_ = model_name
            
        if path is None:
            self.path = './euler_models/eulernd1/'+model_name_+'/'
        else:
            self.path = path+'/'+model_name+'/'
                
    def call(self, x, training=False):
        Ux = self.U(x)
        Ux = self.U_act(Ux)
        Vx = self.V(x)
        Vx = self.V_act(Vx)
        x = self.base_layers[0](x)
        x = self.act_layers[0](x)
        for base_layer,act_layer in zip(self.base_layers[1:], self.act_layers[1:]):
            x = base_layer(x)
            x = act_layer(x)
            x = (1.-x)*Ux + x*Vx

        return self.output_layer(x)


    def loss(self, vd, vp, sc):
        return self.loss_deriv(vd,vp,sc)

    @staticmethod
    def loss_MSE(residual):
        return tf.reduce_mean(tf.square(residual))

    @staticmethod
    def loss_logcosh(residual):
        return tf.reduce_mean(tf.math.log(tf.math.cosh(residual)))

    @tf.function
    def loss_deriv(self, vec_data, vec_pred, space_coll):
        #U_coll = []
        dspace = []

        if self.do_visc:
            rho,v,p,drho,dv,dp,d2v = self.derivs_w_visc(space_coll)
        else:
            rho,v,p,drho,dv,dp = self.derivs(space_coll)

        # data/boundary MSE
        # avg it by number of points, not samples
        # i.e. sum(rho diff**2)/np + sum(vx diff**2)/np etc.
        # and not sum(diff**2)/(np*8), which is what loss_MSE does
        l0 = self.loss_type(vec_data-vec_pred)*self.output_dim
        
        # d/dx: idx 0, d/dt: idx 1
        #dU[:,idx]

        #continuity eq: drho/dt + v*drho/dx + rho*dv/dx = 0
        # and we want d/dt + d/dx = 0
        space_deriv = drho[:,0]*v + dv[:,0]*rho
        time_deriv = drho[:,1]

        l1 = self.loss_type(time_deriv+space_deriv)

        # vx eq:
        #issues with rho = 0, so multiply by rho to get
        # rho*dv/dt + rho*v*dv/dx + dp/dx = 0
        space_deriv = dv[:,0]*v*rho + dp[:,0]
        time_deriv = rho*dv[:,1]

        # -nu*d2u/dx2 term
        if self.do_visc:
            space_deriv = space_deriv - self.visc*rho*d2v[:]

        l2 = self.loss_type(time_deriv+space_deriv)

        # pressure eq: dp/dt + gamma*p*dv/dx + v*dp/dx
        space_deriv = tf.scalar_mul(self.gamma,dv[:,0]*p) + dp[:,0]*v
        time_deriv = dp[:,1]

        l3 = self.loss_type(time_deriv+space_deriv)

        # note that tape.gradient(losses) will sum over
        #  all dloss/dparam for loss in losses
        return [l0,l1,l2,l3]

    # without viscosity
    def derivs(self, space_coll):
        dx = self.dx
        dt = self.dt

        U_coll = self(space_coll, training=True)
        U_coll_px = self(space_coll+self.dx_arr, training=True)
        U_coll_mx = self(space_coll-self.dx_arr, training=True)
        U_coll_pt = self(space_coll+self.dt_arr, training=True)
        U_coll_mt = self(space_coll-self.dt_arr, training=True)

        rho = U_coll[:,0]
        v = U_coll[:,1]
        p = U_coll[:,2]

        drho = tf.stack([(U_coll_px[:,0] - U_coll_mx[:,0])/(2*dx), (U_coll_pt[:,0] - U_coll_mt[:,0])/(2*dt)],axis=1)
        dv = tf.stack([(U_coll_px[:,1] - U_coll_mx[:,1])/(2*dx), (U_coll_pt[:,1] - U_coll_mt[:,1])/(2*dt)],axis=1)
        dp = tf.stack([(U_coll_px[:,2] - U_coll_mx[:,2])/(2*dx), (U_coll_pt[:,2] - U_coll_mt[:,2])/(2*dt)],axis=1)

        return rho,v,p,drho,dv,dp

    # with viscosity for velocity
    def derivs_w_visc(self,space_coll):
        dx = self.dx
        dt = self.dt

        U_coll = self(space_coll, training=True)
        U_coll_px = self(space_coll+self.dx_arr, training=True)
        U_coll_mx = self(space_coll-self.dx_arr, training=True)
        U_coll_pt = self(space_coll+self.dt_arr, training=True)
        U_coll_mt = self(space_coll-self.dt_arr, training=True)

        rho = U_coll[:,0]
        v = U_coll[:,1]
        p = U_coll[:,2]

        drho = tf.stack([(U_coll_px[:,0] - U_coll_mx[:,0])/(2*dx), (U_coll_pt[:,0] - U_coll_mt[:,0])/(2*dt)],axis=1)
        dv = tf.stack([(U_coll_px[:,1] - U_coll_mx[:,1])/(2*dx), (U_coll_pt[:,1] - U_coll_mt[:,1])/(2*dt)],axis=1)
        dp = tf.stack([(U_coll_px[:,2] - U_coll_mx[:,2])/(2*dx), (U_coll_pt[:,2] - U_coll_mt[:,2])/(2*dt)],axis=1)

        # only x derivative is used
        d2v = (U_coll_px[:,1] - 2*v + U_coll_mx[:,1])/(dx*dx)

        return rho,v,p,drho,dv,dp,d2v

    def train(self, space_vec, data_vec, n_coll_dim, space_range, time_range, epochs=80000, randomize_ep=1, lr_decay=0.75, warmup_eps=0, nc_prog=None, anneal_eps = 8500):
        """
        space_vec: input location tensor; shape (n_pts, 2); for bounding positions in spacetime
        data_vec : input data tensor; shape (n_pts, n_output_dim); for bounding plasma data
        n_coll_dim : input N; used to generate N^2 random collocation points
                     (low number, e.g. 20, recommended for warmup period)
        space_range : list of lists; [[x_left, x_right]]; space extent of reconstruction domain
        time_range : list; [t_0, t_1]; time extent of reconstruction domain
        epochs : maximum number of training steps
        randomize_ep : how often to randomize collocation point distribution
        lr_decay : factor to reduce learning rate after last collocation points reached
        warmup_eps : number of training steps to "warmup" network; 
                     training schedule starts after this period
        nc_prog : list; denotes increments of n_coll_dim of training schedule
        anneal_eps : duration of each training period with each n_coll_dim in nc_prog
        
        """

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.path+'/log_'+current_time
        os.makedirs(self.log_path)
        self.summary_writer = tf.summary.create_file_writer(self.log_path)

        progress = []

        spacetime_coloc = generate_spacetime_coloc_rand(space_range, time_range, n_coll_dim, tensor=True)
        spacetime_coloc = tf.concat((spacetime_coloc, space_vec),axis=0)

        template = 'Epoch {}, Data Loss: {}, Physics Loss: {}'

        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(lr=1e-3, epsilon=1e-4)

        lr_holder = self.optimizer.learning_rate.numpy()
        lr_holder_orig = lr_holder
        itw = 0
        ckpt_no = 0
        anneal = True

        ftname = self.log_path+'/timing_notes.txt'
        with open(ftname, 'w') as ftiming:
            ftiming.write("Timings, start ncoll: {}\n".format(n_coll_dim))

        start = timer()
        for epoch in range(1,epochs):

            with tf.GradientTape() as tape:
                predict_vec = self(space_vec, training = True)
                losses = self.loss(data_vec, predict_vec, spacetime_coloc)

            grad_loss = tape.gradient(losses, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad_loss, self.trainable_variables))

            #metrics
            with self.summary_writer.as_default():
                tf.summary.scalar('data loss', losses[0], step=epoch)
                tf.summary.scalar('phys loss', sum(losses[1:]), step=epoch)
                tf.summary.scalar('cont loss', losses[1], step=epoch)
                tf.summary.scalar('v loss', losses[2], step=epoch)
                tf.summary.scalar('p loss', losses[3], step=epoch)

            if(epoch < 10):
                print(template.format(epoch, losses[0].numpy(), tf.math.reduce_sum(losses[1:]).numpy()))
            if(epoch%1000 == 0):
                print(template.format(epoch, losses[0].numpy(), tf.math.reduce_sum(losses[1:]).numpy())+" nc:{}".format(n_coll_dim))

            if(epoch%randomize_ep == 0):
                # reset coloc points
                spacetime_coloc = generate_spacetime_coloc_rand(space_range, time_range, n_coll_dim, tensor=True)
                spacetime_coloc = tf.concat((spacetime_coloc, space_vec),axis=0)

            if(epoch%5000==0):
                self.save_self(self.log_path+'/ckpt{}_weights/'.format(ckpt_no))
                ckpt_no += 1
                print('checkpoint saved')
                end = timer()
                delta_t = timedelta(seconds=end-start)
                with open(ftname, 'a') as ftiming:
                    ftiming.write("epoch: {}, elapsed time: {}\n".format(epoch, delta_t))
                start = timer()

            if sum(losses).numpy() < self.min_loss:
                self.min_loss = sum(losses).numpy()
                self.save_self(self.log_path+'/best_weights/')

            if(epoch > warmup_eps):
                #start annealing
                if(itw == 0):
                    n_coll_dim = nc_prog[itw]
                    itw += 1
                    with open(ftname, 'a') as ftiming:
                        ftiming.write("n coll dim now: {}\n".format(n_coll_dim))

                if((epoch-warmup_eps)%1000 == 0):
                    #check for stagnation
                    print("loss check", self.min_loss, self.ckpt_min_loss, (self.min_loss-self.ckpt_min_loss)/self.ckpt_min_loss)
                    stag_check = (self.ckpt_min_loss-self.min_loss)/self.ckpt_min_loss
                    self.ckpt_min_loss = self.min_loss

                # Annealing: increase density of colocation points according to
                # predefined annealing schedule
                if(anneal and ((epoch-warmup_eps)+1)%anneal_eps == 0):
                    self.load_weights(self.log_path+'/best_weights/weights')
                    if itw < len(nc_prog):
                        # go to next ncoll in progression
                        n_coll_dim = nc_prog[itw]
                        itw += 1
                        with open(ftname, 'a') as ftiming:
                            ftiming.write("n coll dim now: {}\n".format(n_coll_dim))
                        print("n coll dim now", n_coll_dim)

                    else:
                        #end of progression; decrease lr
                        lr_holder = lr_holder * lr_decay
                        self.optimizer.learning_rate = lr_holder
                        with open(ftname, 'a') as ftiming:
                            ftiming.write("learning rate now {}\n".format(lr_holder))
                        print("learning rate now {}".format(lr_holder))

                    if lr_holder / lr_holder_orig < 0.5:
                        print("learning rate minimum hit; ending training")
                        #anneal = False
                        break


        end = timer()
        delta_t = timedelta(seconds=end-start)
        with open(ftname, 'a') as ftiming:
            ftiming.write("epoch: {}, elapsed time: {}\n".format(epoch, delta_t))
        return 0

    def save_self(self, path):
        self.save_weights(path+'weights')

    # warning: does not automatically load optimizer state
    # what we need to do is to run new model (with same architecture)
    # for some steps (1 or 2 is sufficient)
    # and then we can call this to load the full state (including optimizer)
    def load_self(self, path):
        self.load_weights(path+'weights')
