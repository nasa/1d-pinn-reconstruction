# MHD_NN_reconstruction

This code is associated with the paper "Neural Network Reconstruction of Plasma Space-Time" by C.Bard and J. Dorelli (DOI: 10.3389/fspas.2021.732275)

Jupyter notebooks for reproducing paper plots are found in ./reproduce_plots ; the main programs are found in `run_euler.py` and `run_mhd_recon.py`. 

The 'model' folders contain the best trained weights for each network. The 'spacetime' folders contain the raw HDF5 data from the baseline simulations.

Package Requirements: tensorflow, numpy, scipy, pylab, h5py 


If you would like to cite this paper, the BibTex format is:

       @ARTICLE{2021FrASS...8..146B,
       author = {{Bard}, C. and {Dorelli}, J.~C.},       
        title = "{Neural Network Reconstruction of Plasma Space-Time}",        
      journal = {Frontiers in Astronomy and Space Sciences},      
     keywords = {Space Physics, reconstruction, Physics-informed neural network, MHD, computational methods},     
         year = 2021,         
        month = sep,        
       volume = {8},       
          eid = {146},          
        pages = {146},        
          doi = {10.3389/fspas.2021.732275},          
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021FrASS...8..146B},       
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}      
       }
