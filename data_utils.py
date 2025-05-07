import os
import sys
import h5py
import numpy as np
import pandas as pd

def get_cms_data( num_events_per_class=20000, add_one_hot=False, pt_norm_method = "max_per_category", log_pt=False, drop_angles=False ):

    # paths
    sm_file = "/home/b/Physics/snn-anomalies/data/background_for_training.h5"
    h0_file = "/home/b/Physics/snn-anomalies/data/hToTauTau_13TeV_PU20_filtered.h5"
    hc_file = "/home/b/Physics/snn-anomalies/data/hChToTauNu_13TeV_PU20_filtered.h5"
    a4l_file = "/home/b/Physics/snn-anomalies/data/Ato4l_lepFilter_13TeV_filtered.h5"
    lq_file = "/home/b/Physics/snn-anomalies/data/leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5"

    # load data
    with h5py.File(sm_file, 'r') as file:
        sm_data = file['Particles'][:num_events_per_class,:,:-1]
        np.random.shuffle(sm_data)
    with h5py.File(h0_file, 'r') as file:
        h0_data = file['Particles'][:num_events_per_class,:,:-1]
        np.random.shuffle(h0_data)
    with h5py.File(hc_file, 'r') as file:
        hc_data = file['Particles'][:num_events_per_class,:,:-1]
        np.random.shuffle(hc_data)
    with h5py.File(a4l_file, 'r') as file:
        a4l_data = file['Particles'][:num_events_per_class,:,:-1]
        np.random.shuffle(a4l_data)
    with h5py.File(lq_file, 'r') as file:
        lq_data = file['Particles'][:num_events_per_class,:,:-1]
        np.random.shuffle(lq_data)

    print( "sm data shape before any preprocessing: "+str(sm_data.shape))
    
    # function to one-hot encode
    def convert_to_one_hot( data ):
        data_one_hot = []
        for ev in data:
            ev_one_hot = []
            for i, p in enumerate(ev): 
                if p[0]>0:
                    if i==0:
                        ev_one_hot.append( [1,0,0,0] )
                    if (i>0) and (i<5):
                        ev_one_hot.append( [0,1,0,0] )
                    if (i>=5) and (i<9):
                        ev_one_hot.append( [0,0,1,0] )
                    if (i>=9):
                        ev_one_hot.append( [0,0,0,1] )
                else:
                    ev_one_hot.append( [0,0,0,0] )    
            ev_one_hot = np.array( ev_one_hot )
            data_one_hot.append( ev_one_hot )
        data_one_hot = np.array( data_one_hot )
        data_one_hot = np.concatenate( [ data, data_one_hot ], axis=-1 )
        return data_one_hot

    # one-hot encode the pids
    if add_one_hot:
        sm_data = convert_to_one_hot( sm_data )
        h0_data = convert_to_one_hot( h0_data )
        hc_data = convert_to_one_hot( hc_data )
        a4l_data = convert_to_one_hot( a4l_data )
        lq_data = convert_to_one_hot( lq_data )
        print( "sm data shape after one-hot encoding pids: "+str(sm_data.shape))

    # eta -> eta/4
    sm_data[:,:,1] = sm_data[:,:,1]/4
    h0_data[:,:,1] = h0_data[:,:,1]/4
    hc_data[:,:,1] = hc_data[:,:,1]/4
    a4l_data[:,:,1] = a4l_data[:,:,1]/4
    lq_data[:,:,1] = lq_data[:,:,1]/4
    # phi -> phi/pi
    sm_data[:,:,2] = sm_data[:,:,2]/np.pi
    h0_data[:,:,2] = h0_data[:,:,2]/np.pi
    hc_data[:,:,2] = hc_data[:,:,2]/np.pi
    a4l_data[:,:,2] = a4l_data[:,:,2]/np.pi
    lq_data[:,:,2] = lq_data[:,:,2]/np.pi
    # mean pt in dataset
    if pt_norm_method == "mean_all_constits":
        all_sm_pts = sm_data[:,:,0]
        non_zero_sm_pts = all_sm_pts[ all_sm_pts!=0 ]
        pt_norm = np.mean(non_zero_sm_pts) if non_zero_sm_pts.size > 0 else 0
        sm_data[:,:,0] = sm_data[:,:,0] / pt_norm
        h0_data[:,:,0] = h0_data[:,:,0] / pt_norm
        hc_data[:,:,0] = hc_data[:,:,0] / pt_norm
        a4l_data[:,:,0] = a4l_data[:,:,0] / pt_norm
        lq_data[:,:,0] = lq_data[:,:,0] / pt_norm
    elif pt_norm_method == "max_per_category":
        for i in range(sm_data.shape[1]):
            pt_norm = np.max( sm_data[:,i,0] )
            if pt_norm>0:
                sm_data[:,i,0] = sm_data[:,i,0] / pt_norm
                h0_data[:,i,0] = h0_data[:,i,0] / pt_norm
                hc_data[:,i,0] = hc_data[:,i,0] / pt_norm
                a4l_data[:,i,0] = a4l_data[:,i,0] / pt_norm
                lq_data[:,i,0] = lq_data[:,i,0] / pt_norm
    if pt_norm_method == "mean_per_category":
        for i in range(sm_data.shape[1]):
            all_sm_pts = sm_data[:,i,0]
            non_zero_sm_pts = all_sm_pts[ all_sm_pts!=0 ]
            pt_norm = np.mean(non_zero_sm_pts) if non_zero_sm_pts.size > 0 else 0
            if pt_norm>0:
                sm_data[:,i,0] = sm_data[:,i,0] / pt_norm
                h0_data[:,i,0] = h0_data[:,i,0] / pt_norm
                hc_data[:,i,0] = hc_data[:,i,0] / pt_norm
                a4l_data[:,i,0] = a4l_data[:,i,0] / pt_norm
                lq_data[:,i,0] = lq_data[:,i,0] / pt_norm
    
    if log_pt:
        sm_data[:,:,0] = np.log( 1+sm_data[:,:,0] ) 
        h0_data[:,:,0] = np.log( 1+h0_data[:,:,0] ) 
        hc_data[:,:,0] = np.log( 1+hc_data[:,:,0] ) 
        a4l_data[:,:,0] = np.log( 1+a4l_data[:,:,0] ) 
        lq_data[:,:,0] = np.log( 1+lq_data[:,:,0] ) 

    if drop_angles:
        # delete eta
        sm_data = np.delete(sm_data, [1], axis=2)
        h0_data = np.delete(h0_data, [1], axis=2)
        hc_data = np.delete(hc_data, [1], axis=2)
        a4l_data = np.delete(a4l_data, [1], axis=2)
        lq_data = np.delete(lq_data, [1], axis=2)
        # and again to get phi
        sm_data = np.delete(sm_data, [1], axis=2)
        h0_data = np.delete(h0_data, [1], axis=2)
        hc_data = np.delete(hc_data, [1], axis=2)
        a4l_data = np.delete(a4l_data, [1], axis=2)
        lq_data = np.delete(lq_data, [1], axis=2)
        print( "sm data shape after dropping angles: "+str(sm_data.shape))

    # flatten
    sm_data = sm_data.reshape(sm_data.shape[0], -1)
    h0_data = h0_data.reshape(h0_data.shape[0], -1)
    hc_data = hc_data.reshape(hc_data.shape[0], -1)
    a4l_data = a4l_data.reshape(a4l_data.shape[0], -1)
    lq_data = lq_data.reshape(lq_data.shape[0], -1)
    print( "sm data shape after flattening: "+str(sm_data.shape))
    
    return { "sm":sm_data, "h0":h0_data, "hc":hc_data, "a4l":a4l_data, "lq":lq_data }



def get_top_data( num_jets=40000, rescale=1 ):

    # paths
    test_imgs_path = "/home/b/Physics/top-tagging-dataset/test_img.h5"
    train_imgs_path = "/home/b/Physics/top-tagging-dataset/train_img.h5"

    # load the data
    df = pd.read_hdf( train_imgs_path, 'table', start=0, stop=num_jets )
    labels = df.loc[:,'is_signal_new'].to_numpy()
    imgs = df.to_numpy()[:,0:1600]*rescale
    qcd_imgs = imgs[np.where(labels==0)]
    top_imgs = imgs[np.where(labels==1)]

    return { "qcd":qcd_imgs, "top":top_imgs }