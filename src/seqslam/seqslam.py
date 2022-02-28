'''
Filename sequence.py
Author Maxtom
'''
import os
import pdb
import time
import ntpath
import cv2
from glob import glob
from scipy.optimize import curve_fit
import pylab
import numpy as np
import pathos.pools as pp

np.seterr(divide='ignore', invalid='ignore')

from src.seqslam.particle   import MSPF
from src.util               import Cosine, load_image, log, save_matches, save_PR, eval_auc, Euclidean
from src.util               import extract_image_feature, getANN
from config                 import cfg

import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab

plt.style.use('ggplot')

class SeqSlam(object):
    """ SeqSLAM method """
    def __init__(self):
        """ Seqslam framework. """
        log("Current method {}".format(cfg.method), mode="b")
        self.m_diff = []
        self.m_enhance = []
        self.matches = []
    def run(self, im_test, im_ref):
        """ Run Seqslam framework. """
        # test path
        test = glob(os.path.join(im_test, cfg.synax))
        ref = glob(os.path.join(im_ref, cfg.synax))
        test.sort()
        ref.sort()
        test = test[::-cfg.init_skip]
        ref = ref[::-cfg.init_skip]
        test = test[::-1]
        ref = ref[::-1]
        log("Test data len {}".format(len(test)), mode='g')
        log("Refer data len {}".format(len(ref)), mode='g')

        startT = time.time()
        self.m_diff, self.m_enhance, self.matches = estimate_track_SeqSLAM(test, ref)
        log("Total matching time {}".format(time.time()-startT), mode='y')
        #print (self.matches)
        #pdb.set_trace()
        eval_track_detail(self.m_diff, self.m_enhance, self.matches)
        
def estimate_track_MSPF_SeqSLAM(test_feature, ref_feature, p_id):
    """ Estimate track """
    # caculate image difference matrix
    #startT = time.time()
    m_diff = get_difference_matrix(test_feature, ref_feature)
    #log("Get difference, used time {} for {}".format(time.time()-startT, p_id), mode='g')
    # caculate enhancement matrix
    #startT = time.time()
    m_enhance = get_enhancement_matrix(m_diff)
    #log("Get enhancement, used time {} for {}".format(time.time()-startT, p_id), mode='b')
    # caculate potential matches
    #startT = time.time()
    matches = get_matches(m_enhance)
    #log("Get matches, used time {} for {}".format(time.time()-startT, p_id), mode='y')
    return m_diff, m_enhance, matches
def estimate_track_SeqSLAM(im_test, im_ref):
    """ Estimate track """
    # begin image processing
    startT = time.time()
    log("Begin Estimation", mode="r")
    test_feature = get_features(im_test)
    ref_feature = get_features(im_ref)
    log("Done preprocessing, used time {}".format(time.time()-startT), mode='g')
    # caculate image difference matrix
    startT = time.time()
    m_diff = get_difference_matrix(test_feature, ref_feature)
    log("Done difference, used time {}".format(time.time()-startT), mode='g')
    # caculate enhancement matrix
    startT = time.time()
    m_enhance = get_enhancement_matrix(m_diff)
    log("Done enhancement, used time {}".format(time.time()-startT), mode='g')
    # caculate potential matches
    startT = time.time()
    matches = get_matches(m_enhance)
    log("Done matching, used time {}".format(time.time()-startT), mode='g')
    return m_diff, m_enhance, matches
def eval_track_detail(m_diff, m_enhance, matches):
    """ Evaluate detail tracking """
    # Evaluate matching reslut
    log("Begin Evaluation", mode="r")
    # Save matches figure
    save_matches(m_diff, m_enhance, matches, "seqslam")
    log("Done saving matches", mode='g')
    # Save PR_Curve
    save_PR(matches)
def get_features(data_path, show=False):
    """ Get Features """
    # load images
    features = np.zeros((len(data_path), cfg.img_size*cfg.img_size))
    for count, addr in enumerate(data_path):
        if show:
            print (addr)
        feature = load_image(addr)
        features[count, :] = feature
    return features
def get_difference_matrix(ref, test):
    """ Get difference matrix """
    diff = Cosine(ref, test)
    diff = np.exp(1-diff)
    diff = Euclidean(ref, test)
    return diff
def get_enhancement_matrix(diff):
    """ Evalute enhancement matrix """
    ddiff = np.zeros(diff.shape)
    for i in range(diff.shape[0]):
        a=np.max((0, i-cfg.constrastEnhancementR/2))
        b=np.min((diff.shape[0], i+cfg.constrastEnhancementR/2+1))
        v = diff[a:b, :]
        ddiff[i,:] = (diff[i,:] - np.mean(v, 0)) / np.std(v, 0, ddof=1)  
    return ddiff-np.min(np.min(ddiff))
def get_matches(diff):
    """ Get potential matches """
    # Load parameters
    v_ds = cfg.v_ds
    vmax = cfg.vmax
    vmin = cfg.vmin
    #Rwindow = cfg.Rwindow
    matches = np.nan*np.ones((diff.shape[1], 3))
    # parfor?
    for N in range(int(v_ds/2), diff.shape[1]-int(v_ds/2)):
        move_min = vmin * v_ds
        move_max = vmax * v_ds
        move = np.arange(int(move_min), int(move_max)+1)
        v = move.astype(float) / v_ds
        idx_add = np.tile(np.arange(0, v_ds+1), (len(v),1))
        idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T)
        # this is where our trajectory starts
        n_start = N + 1 - v_ds/2
        x = np.tile(np.arange(n_start , n_start+v_ds+1), (len(v), 1))
        #TODO idx_add and x now equivalent to MATLAB, dh 1 indexing
        score = np.zeros(diff.shape[0])
        # add a line of inf costs so that we penalize running out of data
        diff = np.vstack((diff, np.infty*np.ones((1, diff.shape[1]))))
        y_max = diff.shape[0]
        xx = (x-1) * y_max
        flatDD = diff.flatten(1)
        for s in range(1, diff.shape[0]):
            y = np.copy(idx_add+s)
            y[y > y_max] = y_max
            idx = (xx + y).astype(int)
            ds = np.sum(flatDD[idx-1], 1)
            score[s-1] = np.min(ds)
        min_idx = np.argmin(score)
        min_value = score[min_idx]
        match = [N, min_idx + v_ds/2, 1. / min_value]
        if match[2] > 1:
            match[2] = 1.0
        matches[N, :] = match
    return matches