'''
Filename sequence.py
Author Maxtom
'''
import os
import pdb
import time
import ntpath
import cv2
import Image
from glob import glob
from scipy.optimize import curve_fit
import pylab
import numpy as np
import pathos.pools as pp
np.seterr(divide='ignore', invalid='ignore')
from src.seqslam.particle import MSPF
from src.util import Cosine, load_image, log, save_matches, save_PR, eval_auc, Euclidean
from src.util import extract_image_feature, getANN
from config import cfg
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab
plt.style.use('ggplot')

font = {'family' : 'normal',
        'size'   : 22}
        #'weight' : 'bold',

matplotlib.rc('font', **font)

class MspfSeqSlam(object):
    """ MSPF SeqSLAM method """
    def __init__(self):
        """ MSPF Seqslam framework. """
        log("Current method {}".format(cfg.method), mode="b")
        self.matches = []
        self.mspf = []
        self.test_feature = []
        self.ref_feature = []
    @staticmethod
    def get_difference(cur, refs):
        """ Extract Sift feature to compute between cur to reference images. """
        result, distance = getANN(refs, cur, cfg.pf_num)
        return result[0], distance
    @staticmethod
    def extract_track(test_imgs, ref_imgs, index):
        """ Extract potential track based on the index. """
        test = test_imgs
        test_len = len(test_imgs)
        if index > test_len:
            id_begin = index-test_len
            id_end = index
        else:
            id_begin = 0
            id_end = test_len
        ref = ref_imgs[id_begin:id_end]
        return test, ref
    def update_particle_id(self, old_level, new_level):
        for particle in self.mspf.particles:
            particle.index = new_level - (old_level - particle.index)*2
            if particle.index < 0 or particle.index >= new_level:
                log("error index for particle {}".format(particle.index), 'r')
    def extract_valid_index(self, match, index, is_plot=False):
        """ Based on current matches, extract new valid index. """
        index_len = len(match)
        match = match[~np.isnan(match)]
        match = np.reshape(match, (len(match)/3, 3))
        res = []
        #res_len = []
        difference = match[:, 0]-match[:, 1]
        for x in range(-(int)(index_len/2), (int)(index_len/2)):
            value = np.array(abs(match[:, 0]-(match[:, 1]+x)))
            value = value*np.array(match[:,2])
            #eff_len = len(value[value<0.2*index_len])
            #value[value>0.2*index_len] = 100
            #if  eff_len > 0.4*index_len:
            #    value *= 1.0 #(x+(int)(index_len/2))
            #else:
            #    value += 10000
            res.append(value.sum())
            #res_len.append(eff_len)
        #res_mean = np.array(res).mean(1)
        #res_value = 1./(np.min(res_mean)+1.0)
        #res_index = np.argmin(res_mean)-(int)(index_len/2)
        res_index = np.argmin(res)-(int)(index_len/2)
        res_value = 1./(np.min(res)+1.0)
        fig = []
        if is_plot:
            fig = pylab.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(match[:, 0], match[:, 1], 'o', label='Original')
            ax.plot(match[:, 0], match[:, 0]-res_index, label='Line')
            ax.legend(loc='lower center')
        index_shift = res_index
        return res_value, index_shift, fig
    def particle_measure_update(self, p_id, is_plot=False):
        particle = self.mspf.particles[p_id]
        tempT = time.time()
        cur_test, cur_ref = self.extract_track(self.ds_test, self.ds_ref, particle.index)
        D, DD, match = estimate_track_MSPF_SeqSLAM(cur_test, cur_ref, p_id)
        weight, index_shift, fig = self.extract_valid_index(match, particle.index, is_plot)
        #log("processing for {} with {}".format(p_id, time.time()-tempT), mode="g")
        if is_plot==False:
            self.mspf.particles[p_id].index += index_shift
            self.mspf.particles[p_id].weight *= weight
            return self.mspf.particles[p_id].index, self.mspf.particles[p_id].weight
        else:
            return fig, match, index_shift
    def cache_data(self, im_test, im_ref):
        test = glob(os.path.join(im_test, cfg.synax))
        ref = glob(os.path.join(im_ref, cfg.synax))
        test.sort()
        ref.sort()
        log("Test data len {}".format(len(test)), mode='g')
        log("Refer data len {}".format(len(ref)), mode='g')
        # Create refer image
        test_name = im_test+".npy"
        if os.path.exists(test_name):
            self.test_feature = np.load(test_name)
            log("Load "+test_name, mode='r')
        else:
            self.test_feature = get_features(test, show=True)
            np.save(test_name, self.test_feature)
            log("Save "+test_name, mode='y')
        ref_name = im_ref+".npy"
        if os.path.exists(ref_name):
            self.ref_feature = np.load(ref_name)
            log("Load "+ref_name, mode='r')
        else:
            self.ref_feature = get_features(ref, show=True)
            np.save(ref_name, self.ref_feature)
            log("save "+ref_name, mode='y')
    def call_index(self, im_test, im_ref, resolution, index):
        test = glob(os.path.join(im_test, cfg.synax))
        ref0 = glob(os.path.join(im_ref, cfg.synax))
        test.sort()
        ref0.sort()
        ref = ref0[::-resolution]
        ref = ref[::-1]
        return test[-1], ref[index]
    def down_sampling(self, resolution):
        """
        Multi-scale particle filter based Feature Extraction.
        Ref:  *****************************************************
        Down: *   *   *   *   *   *   *   *   *   *   *   *   *   *
        Test:                          ********
        TD:                               *   *
        Commit: down sample images, improve searching speed.
        """
        test = self.test_feature[::-resolution]
        ref = self.ref_feature[::-resolution]
        test = test[::-1]
        ref = ref[::-1]
        log("Refer len is {}".format(len(ref)), mode="b")
        log("Test len is {}".format(len(test)), mode="b")
        return test, ref
    def run(self, im_test, im_ref):
        """ MSPF running framework """
        # Cache Data
        self.cache_data(im_test, im_ref)
        # Down sampling track based on original resolution level
        re_level = cfg.data_skip
        self.ds_test, self.ds_ref = self.down_sampling(re_level)
        #cfg.pf_num = 400
        cfg.pf_num = int(cfg.tau*len(self.ds_ref)/len(self.ds_test))
        log("current pf number is {}".format(cfg.pf_num), mode='r')
        p_ids = np.linspace(len(self.ds_test), len(self.ds_ref), \
                            num=cfg.pf_num, endpoint=True, dtype=int)
        self.mspf = MSPF(cfg.pf_num)
        self.mspf.particle_sampling(p_ids)
        # Coarse to fine particle filte
        run_time = time.time()
        for dep in range(cfg.max_depth):
            for pfit in range(cfg.pf_iter):
                self.t_name = "dep_{}_pts_{}".format(dep, pfit)

                log("Update for particle", mode='r')

                # Updateing Weighint
                statT = time.time()
                #jobs = [p_id for p_id in range(len(self.mspf.particles))]
                #p = pp.ProcessPool(1)
                #result = p.map(self.particle_measure_update, jobs)
                indexs = []
                weights = []
                for p_id in range(len(self.mspf.particles)):
                    index, weight = self.particle_measure_update(p_id)
                    indexs.append(index)
                    weights.append(weight)

                result = zip(indexs, weights)
                log("Matching time used {}".format(time.time() - statT), mode='g')

                # Particle Updating
                statT = time.time()
                particle_dist = int((len(self.ds_ref)-len(self.ds_test))/len(self.mspf.particles))
                ef_pf_rate = self.mspf.updating(result, particle_dist)
                log("Paritcle Update used {}".format(time.time() - statT), mode='g')



                # Value current best match
                self.mspf.best_estimate()
                test_index, ref_index = self.call_index(im_test, im_ref, \
                                                    re_level, self.mspf.particles[0].index)
                log("test {}".format(test_index), mode='g')
                log("refer {}".format(ref_index), mode='g')
                sum_weights = []
                for idx in range(min(len(self.mspf.particles), 20)):
                     sum_weights.append(self.mspf.particles[idx].weight)

                '''
                for idx in range(min(len(self.mspf.particles), 20)):

                    ## show some results
                    plt.figure()
                    x = np.arange(min(len(self.mspf.particles), 20))
                    plt.xlim(-1, min(len(self.mspf.particles), 20))
                    plt.bar(x, sum_weights, align='center', alpha=0.5)
                    plt.bar(idx, self.mspf.particles[idx].weight, color='yellow', align='center', alpha=0.5)
                    plt.grid(True)
                    plt.savefig(os.path.join(cfg.log_dir, 'bar.jpg'))
                    plt.close()

                    print (self.mspf.particles[idx].weight)
                    try:
                        test_index, ref_index = self.call_index(im_test, im_ref, \
                                                    re_level, self.mspf.particles[idx].index)
                    except:
                        log("index {} out of range".format(self.mspf.particles[idx].index), mode='y')
                        continue

                    log("refer {}'s index is {} ".format(idx, ref_index), mode='b')
                    fig, match, new_index = self.particle_measure_update(idx, is_plot=True)
                    fig.savefig(os.path.join(cfg.log_dir, self.t_name+"{}.jpg".format(idx)))
                    img1 = cv2.imread(test_index)
                    img2 = cv2.imread(ref_index)
                    img3 = cv2.imread(os.path.join(cfg.log_dir, self.t_name+"{}.jpg".format(idx)))
                    img4 = cv2.imread(os.path.join(cfg.log_dir, "bar.jpg"))
                    img1 = cv2.resize(img1, (320, 180))
                    img2 = cv2.resize(img2, (320, 180))
                    img3 = cv2.resize(img3, (320, 180))
                    img4 = cv2.resize(img4, (320, 180))
                    # save_name = os.path.join(cfg.log_dir, self.t_name+'_particle_{}_refer_{}_1.png'.format(idx, ref_index.split('/')[-1]))
                    # cv2.imwrite(save_name, img1)
                    # save_name = os.path.join(cfg.log_dir, self.t_name+'_particle_{}_refer_{}_2.png'.format(idx, ref_index.split('/')[-1]))
                    # cv2.imwrite(save_name, img2)
                    # save_name = os.path.join(cfg.log_dir, self.t_name+'_particle_{}_refer_{}_3.png'.format(idx, ref_index.split('/')[-1]))
                    # cv2.imwrite(save_name, img3)
                    vis1 = np.concatenate((img1, img2), axis=1)
                    vis2 = np.concatenate((img3, img4), axis=1)
                    # cv2.imwrite(os.path.join(cfg.log_dir, "v1.jpg"), vis1)
                    # cv2.imwrite(os.path.join(cfg.log_dir, "v2.jpg"), vis2)
                    # v_im1 = cv2.imread(os.path.join(cfg.log_dir, "v1.jpg"))
                    # v_im2 = cv2.imread(os.path.join(cfg.log_dir, "v2.jpg"))
                    vis = np.concatenate((vis1, vis2), axis=1)
                    #vis = np.concatenate((img1, img2, img3), axis=1)
                    save_name = os.path.join(cfg.log_dir, self.t_name+'_particle_{}.png'.format(idx))
                    cv2.imwrite(save_name, vis)
                    pylab.close()
                    if idx==0:
                        log("test {}".format(test_index), mode='r')
                        log("refer {}".format(ref_index), mode='r')
                        #os.system('eog {}'.format(test_index))
                        #os.system('eog {}'.format(ref_index))
                        print ("old index {}".format(self.mspf.particles[idx].index))
                        print ("new index {}".format(new_index))
                
                    os.system("rm {}/*.jpg".format(cfg.log_dir))
                # Resampling based on particle distributions
                # if ef_pf_rate <= cfg.th_particle:
                #     #pf_num = int(len(self.mspf.particles)/2)
                #     pf_num = len(self.mspf.particles)
                #     self.mspf.re_sampling(re_level, pf_num)
                #     log("After resampling particle {}".format(len(self.mspf.particles)), mode='y')
                #pdb.set_trace()
                '''
                log("Effective prticles rate {}".format(ef_pf_rate), mode='g')
                if ef_pf_rate <= 0.5:
                    break
                    
            # Change map scale and resolution level
            re_level = int(re_level/2)
            if re_level == 0:
                re_level = 1
            # Update particle ID
            old_ref_len = len(self.ds_ref)
            self.ds_test, self.ds_ref = self.down_sampling(re_level)
            new_ref_len = len(self.ds_ref)
            self.update_particle_id(old_ref_len, new_ref_len)
            '''
            print (" ")
            for idx in range(min(len(self.mspf.particles), 20)):
                try:
                    test_index, ref_index = self.call_index(im_test, im_ref, \
                                                re_level, self.mspf.particles[idx].index)
                except:
                    log("index {} out of range".format(self.mspf.particles[idx].index), mode='y')
                    continue

                log("refer {}'s index is {} ".format(idx, ref_index), mode='b')
            '''
        log("Totoal run time {}".format(time.time()-run_time), mode='r')
        return self.matches

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
'''
def get_better_matches(diff):
    # Load parameters
    v_ds = args.v_ds
    vmax = args.vmax
    vmin = args.vmin
    Rwindow = args.Rwindow
    matches = np.nan*np.ones((DD.shape[1],2))

    # We shall search for matches using velocities between
    # params.matching.vmin and params.matching.vmax.
    # However, not every vskip may be neccessary to check. So we first find
    # out, which v leads to different trajectories:
    
    move_min = vmin * v_ds
    move_max = vmax * v_ds    
        
    # Obtain the v steps,
    # in case vmin = 0.8, vmax = 1.1, v equal to
    #     array([ 0.8,  0.9,  1. ,  1.1])
    move = np.arange(int(move_min), int(move_max)+1)
    v = move.astype(float) / v_ds

    # Obtain the addition (v,ds) matrix,
    # in case of v_ds = 10, and v as above, the idx_add is equal to
    # array([[  0.,   0.,   1.,   2.,   3.,   4.,   4.,   5.,   6.,   7.,   8.],
    #        [  0.,   0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
    #        [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.],
    #        [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  11.]]
    idx_add = np.tile(np.arange(0, v_ds+1), (len(v),1))
    idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T)

    # Obtain the base (v,ds) matrix,
    # in case of the above setting, and N = 100, x is equal to
    # array([[ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106],
    #        [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106],
    #        [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106],
    #        [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106]])
    n_start = 1 - v_ds/2    
    x= np.tile(np.arange(n_start , n_start+v_ds+1), (len(v), 1))  

    # parfor?
    for N in range(v_ds/2, DD.shape[1]-v_ds/2):
        x_n = x+N
        # add a line of inf costs so that we penalize running out of data
        DD=np.vstack((DD, np.infty*np.ones((1,DD.shape[1]))))
        # Extend the base (v,ds) matrix into a flatten based index
        # in case of the data len is 1000, xx is equal to
        # array([[ 95000,  96000,  97000,  98000,  99000, 100000, 101000, 102000, 103000, 104000, 105000],
        #        [ 95000,  96000,  97000,  98000,  99000, 100000, 101000, 102000, 103000, 104000, 105000],
        #        [ 95000,  96000,  97000,  98000,  99000, 100000, 101000, 102000, 103000, 104000, 105000],
        #        [ 95000,  96000,  97000,  98000,  99000, 100000, 101000, 102000, 103000, 104000, 105000]])
        y_max = DD.shape[0]      
        xx = (x_n-1) * y_max
        flatDD = DD.flatten(1)

        print (y_max)
        # Initial Score for K nearest
        score = np.zeros(Ann.shape[1])            

        ## Obtain the score of K nearest points
        for id in range(Ann.shape[1]):
            s = Ann[N, id]
            y = np.copy(idx_add+s)
            y[y>y_max]=y_max     
            idx = (xx + y).astype(int)
            ds = np.sum(flatDD[idx-1],1)
            score[id] = np.min(ds)
            
        # find min score and 2nd smallest score outside of a window
        # around the minimum 
        min_idx   = Ann[N, np.argmin(score)]
        min_value = score[np.argmin(score)]

        a1 = Ann[N,:] > (min_idx + Rwindow/2)
        a2 = Ann[N,:] < (min_idx - Rwindow/2)


        match = [min_idx + v_ds/2, 1. / min_value]
        if match[1] > 1:
            match = 1.0
        if len(score[a1+a2]) > 0:
            min_value_2nd = np.min(score[a1+a2])
            match = [min_idx + v_ds/2, min_value_2nd / min_value]
        else:
            match = [min_idx + v_ds/2, 0.2]
        matches[N,:] = match
    
    return matches
'''
