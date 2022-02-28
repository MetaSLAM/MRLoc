'''
Filename sequence.py
Author Maxtom
'''
import os
import pdb
import time
import ntpath
import cv2
# import Image
from glob import glob
from scipy.optimize import curve_fit
import pylab
import numpy as np
import pathos.pools as pp

from src.util.OrthogonalList import OrthogonalList

np.seterr(divide='ignore', invalid='ignore')
from src.seqslam.particle import MSPF
from src.util import Cosine, load_image, load_feature, log, save_matches, save_PR, eval_auc, Euclidean
from src.util import extract_image_feature, getANN
from config import cfg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

plt.style.use('ggplot')

font = {'family': 'normal',
        'size': 22}
# 'weight' : 'bold',

matplotlib.rc('font', **font)

class MspfSeqSlam(object):
    """ MSPF SeqSLAM method """
    def __init__(self):
        """ MSPF Seqslam framework. """
        log("Current method {}".format(cfg.method), mode="b")
        self.matches        = []
        self.mspf           = []
        self.test_feature   = []
        self.online_feature = []
        self.ref_feature    = []
        self.ref_path       = []
        self.test_path      = []
        self.best_index     = 0
        self.test_count     = 0

        self.ds_test        = [] # Downsampled test dataset
        self.ds_ref         = [] # Downsampled ref dataset
        self.ol = OrthogonalList()

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
        
        # FIXME Only works for short range
        if index > test_len:
            id_begin = index - test_len
            id_end = index
            ref = ref_imgs[id_begin:id_end]
        else:
            # id_begin = 0
            # id_end = test_len
            id_begin = index
            id_begin = test_len-index
            ref = np.concatenate((ref_imgs[0:id_begin], ref_imgs[-id_begin:]), axis=0) 
        
        return test, ref

    def update_particle_id(self, old_level, new_level):
        for particle in self.mspf.particles:
            particle.index = new_level - (old_level - particle.index) * 2
            if particle.index < 0 or particle.index >= new_level:
                log("error index for particle {}".format(particle.index), 'r')
        self.best_index = new_level - (old_level - self.best_index) * 2

    def extract_valid_index(self, match, index, is_plot=False):
        """ Based on current matches, extract new valid index. """
        index_len = len(match)
        match = match[~np.isnan(match)]
        match = np.reshape(match, (len(match) / 3, 3))
        res = []
        # res_len = []
        difference = match[:, 0] - match[:, 1]
        for x in range(-(int)(index_len / 2), (int)(index_len / 2)):
            value = np.array(abs(match[:, 0] - (match[:, 1] + x)))
            value = value * np.array(match[:, 2])
            res.append(value.sum())
        res_index = np.argmin(res) - (int)(index_len / 2)
        res_value = 1. / (np.min(res) + 1.0)
        fig = []
        if is_plot:
            fig = pylab.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(match[:, 0], match[:, 1], 'o', label='Original')
            ax.plot(match[:, 0], match[:, 0] - res_index, label='Line')
            ax.legend(loc='lower center')
            ax.set_title("residual weighting {}".format(res_value))
        index_shift = res_index
        return res_value, index_shift, fig

    def particle_measure_update(self, p_id, is_plot=False):
        # Step1: Based on the current particle ID and resolution level, 
        #        extract sub-trajectory from reference frames.
        # Step2: Apply sequential matching.

        particle = self.mspf.particles[p_id]

        cur_test, cur_ref = self.extract_track(self.ds_test, self.ds_ref, particle.index)   # find corresponding test and ref

        D, DD, match = estimate_track_MSPF_SeqSLAM(cur_test, cur_ref, p_id)                 # get difference score
        weight, index_shift, fig = self.extract_valid_index(match, particle.index, is_plot) # find the minimum matching
        if is_plot == False:
            self.mspf.particles[p_id].index   += index_shift
            self.mspf.particles[p_id].weight  *= weight
            self.mspf.particles[p_id].residual = weight
            return self.mspf.particles[p_id].index, self.mspf.particles[p_id].weight, weight
        else:
            return fig, DD, index_shift

    def cache_data(self, ref):
        log("Test data len {}".format(len(self.test_path)), mode='g')
        log("Refer data len {}".format(len(self.ref_path)), mode='g')

        # Load testing images
        test_feature = get_features(self.test_path, show=False)
        self.test_feature = test_feature[:cfg.test_start]
        self.online_feature = test_feature[cfg.test_start:]

        # Load refer image
        ref_name = ref + ".npy"
        if os.path.exists(ref_name):
            self.ref_feature = np.load(ref_name)
            log("Load " + ref_name, mode='r')
        else:
            self.ref_feature = get_features(self.ref_path, show=False)
            np.save(ref_name, self.ref_feature)
            log("save " + ref_name, mode='y')

    def call_index(self, resolution, index):
        # TODO check index
        test = self.test_path
        ref0 = self.ref_path
        ref = ref0[::-resolution]
        ref = ref[::-1]

        print (index)
        print (resolution)
        print (len(ref0))
        print (len(test))

        return test[cfg.test_start + self.test_count], ref[index]

    def call_online_index(self, resolution, count, index):
        test = self.test_path
        ref0 = self.ref_path
        ref = ref0[::-resolution]
        ref = ref[::-1]
        new_index = np.mod(index, len(ref))
        return test[cfg.test_start + count], ref[new_index]

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
        ref  = self.ref_feature[::-resolution]
        test = test[::-1]
        ref  = ref[::-1]
        log("Refer len is {}".format(len(ref)), mode="b")
        log("Test len is {}".format(len(test)), mode="b")
        return test, ref

    #####################################################################
    ##     The core module to run multi-resolution paritcle filter     ##
    #####################################################################
    def run(self, im_test, im_ref):

        start_time = time.time()
        # NOTE Step 1: Data Processing
        self.data_processing(im_test, im_ref)
        log("=============================================", mode='g')
        log("Data Processing used time {}".format(time.time()-start_time), mode='r')
        log("=============================================", mode='g')
        print("")

        # NOTE Step 2: Global Searching
        start_time = time.time()
        self.global_search()
        log("=============================================", mode='g')
        log("Global search used time {}".format(time.time()-start_time), mode='r')
        log("=============================================", mode='g')
        print("")

        # NOTE Step 3: Online Searching
        self.online_search()

    #####################################################################
    ##     ROS: global pathching                                       ##
    #####################################################################
    def ros_global_search(self, feature_test, feature_ref):

        # NOTE Step1: Based on the current resolution level, generate data sets
        self.ds_test = feature_test
        self.ds_ref  = feature_ref

        # NOTE Step2: Generate uniform particles along the trajectory
        init_pf = int(cfg.tau * len(self.ds_ref) / len(self.ds_test))
        cfg.pf_num = init_pf
        p_ids = np.linspace(len(self.ds_test), len(self.ds_ref), num=cfg.pf_num, endpoint=True, dtype=int)
        log("current pf number is {}".format(cfg.pf_num), mode='r')

        # NOTE Step3: Initialize Multi Resolution Particle Filter
        self.mspf = MSPF(cfg.pf_num)
        self.mspf.particle_sampling(p_ids)

        # NOTE Step4 Updateing Particles Weighint
        indexs = weights = []
        for p_id in range(len(self.mspf.particles)):
            index, weight, _ = self.particle_measure_update(p_id)
            indexs.append(index)
            weights.append(weight)
        result = zip(indexs, weights)
        particle_dist = int((len(self.ds_ref) - len(self.ds_test)) / len(self.mspf.particles))
        ef_pf_rate = self.mspf.updating(result, particle_dist)

        # NOTE Step5 Estimate best particle
        self.best_index = self.mspf.best_estimate()
        print(self.best_index)

        _, _, measure = self.particle_measure_update(0)

        log("=============================================", mode='g')
        log("Best Particle with measurement {}".format(measure), mode='r')
        log("=============================================", mode='g')
        print("")

        # NOTE Step6: ploting reuslt
        fig, DD, _ = self.particle_measure_update(0, is_plot=True)
        fig.savefig(os.path.join(cfg.log_dir, "matrix.jpg"))
        
        return self.best_index, DD

    #####################################
    ## ANCHOR     Data Processing      ##
    #####################################
    def data_processing(self, im_test, im_ref):
        self.test_path = glob(os.path.join(im_test, cfg.synax))
        self.ref_path = glob(os.path.join(im_ref, cfg.synax))
        self.test_path.sort()
        self.ref_path.sort()
        self.cache_data(im_ref)

        # self.ol.InitData()
        # self.ol.HorizontalLink()
        # self.ol.VerticalLink()
        # self.ol.map()

    #####################################
    ## ANCHOR The Global Search Steps  ##
    #####################################
    def global_search(self):

        # NOTE Step1: Based on the current resolution level, generate data sets
        self.re_level = cfg.data_skip
        self.ds_test, self.ds_ref = self.down_sampling(self.re_level)

        # NOTE Step2: Generate uniform particles along the trajectory
        init_pf = int(cfg.tau * len(self.ref_feature) / len(self.test_feature))
        cfg.pf_num = init_pf
        p_ids = np.linspace(len(self.ds_test), len(self.ds_ref), num=cfg.pf_num, endpoint=True, dtype=int)
        log("current pf number is {}".format(cfg.pf_num), mode='r')

        # NOTE Step3: Initialize Multi Resolution Particle Filter
        self.mspf = MSPF(cfg.pf_num)
        self.mspf.particle_sampling(p_ids)

        # NOTE Step4: Enter the loop of mutli resolution and particle filter estiamtion
        run_time = time.time()

        # NOTE Step4.1 Updateing Particles Weighint
        statT = time.time()
        indexs = weights = []
        for p_id in range(len(self.mspf.particles)):
            index, weight, _ = self.particle_measure_update(p_id)
            indexs.append(index)
            weights.append(weight)
        result = zip(indexs, weights)
        particle_dist = int((len(self.ds_ref) - len(self.ds_test)) / len(self.mspf.particles))
        ef_pf_rate = self.mspf.updating(result, particle_dist)
        log("Paritcle Update used {}".format(time.time() - statT), mode='g')

        # NOTE Step4.2 Estimate best particle
        self.best_index = self.mspf.best_estimate()
        print(self.best_index)
        # test_index, ref_index = self.call_index(self.re_level, self.mspf.particles[0].index)
        # log("Effective prticles rate {}".format(ef_pf_rate), mode='g')
        # log("test {}".format(test_index), mode='g')
        # log("refer {}".format(ref_index), mode='g')
        # sum_weights = []
        # for idx in range(min(len(self.mspf.particles), 20)):
        #     sum_weights.append(self.mspf.particles[idx].weight)

        log("Totoal run time {}".format(time.time() - run_time), mode='r')

    #####################################
    ## ANCHOR The Online Search Steps  ##
    #####################################
    @property
    def online_search(self):
        len_feature = len(self.online_feature)
        new_count = 0
        for count in range(self.test_count, len_feature):
            
            start_time = time.time()
            # NOTE Step1: FIFO for neighbor features
            value = self.online_feature[count]
            if (count % int(self.re_level) == 0):
                self.ds_test = self.ds_test[1:]
                self.ds_test = np.concatenate((self.ds_test, value.reshape(1, -1)), axis=0)
            else:
                continue

            # NOTE Step2: Generate new particles
            test_index, ref_index = self.call_online_index(self.re_level, count, self.best_index)
            search_index = self.best_index * self.re_level
            # log("Test {}, Ref {}".format(test_index, ref_index), mode='r')
            # log("Best index is {}".format(self.best_index),mode='b')
            # FIXME Just use one particle right now
            self.mspf.generate_new_particles(self.best_index, 20, 0) 

            # NOTE Step3: Particles Measurement
            indexs = weights = []
            for p_id in range(len(self.mspf.particles)):
                index, weight, _ = self.particle_measure_update(p_id)
                indexs.append(index)
                weights.append(weight)

            # NOTE Step4: Predicting the best matches
            self.best_index = self.mspf.best_estimate() # sort current particles
            test_index, ref_index = self.call_online_index(self.re_level, count, self.best_index) # find path according to index
    
            # NOTE Step5: check best particle residual
            residual_weight = self.mspf.particles[0].residual
            log("=============================================", mode='g')
            log("Estimate index {} with {}s".format(test_index, time.time()-start_time), mode='r')
            log("=============================================", mode='g')
            print("")

            _, _, measure = self.particle_measure_update(0)

            log("=============================================", mode='g')
            log("Best Particle with measurement {}".format(measure), mode='b')
            log("=============================================", mode='g')
            print("")

            # NOTE Step6: ploting reuslt
            fig, _, _ = self.particle_measure_update(0, is_plot=True)
            fig.savefig(os.path.join(cfg.log_dir, "matrix.jpg"))
            img1 = cv2.imread(test_index)
            img2 = cv2.imread(ref_index)
            img3 = cv2.imread(os.path.join(cfg.log_dir, "matrix.jpg"))
            img1 = cv2.resize(img1, (320 * 2, 180 * 2))
            img2 = cv2.resize(img2, (320 * 2, 180 * 2))
            img3 = cv2.resize(img3, (320 * 2, 180 * 2))
            vis = np.concatenate((img1, img2, img3), axis=1)
            save_name = os.path.join(cfg.log_dir, 'match_{:04}.png'.format(count))
            cv2.imwrite(save_name, vis)

            print("")

































    def generate_searching(self, ref_index):
        item1, item2 = self.ol.detectPath(ref_index)
        print("problem is", ref_index)

        possible_path = []
        possible_index = []
        thresholdL = 20
        thresholdR = 100

        startIs, endIs, core = self.ol.CrossMining(item1, item2)
        for s in startIs:
            for e in endIs:
                print("path: ", s, " -> ", core, " -> ", e)
                # pdb.set_trace()
                search_space = np.concatenate((self.ref_feature[s[0]:s[1]], self.ref_feature[core[0]:core[1]], self.ref_feature[e[0]:e[1]]))
                index_space = np.concatenate((range(s[0], s[1]), range(core[0], core[1]), range(e[0], e[1])))
                num_first_part = s[1] - s[0] + 1
                num_last_part = e[1] - e[0] + 1
                num_core_part = core[1] - core[0] + 1
                center = ref_index - core[0] + num_first_part

                search_space = search_space[center - thresholdL:center + thresholdR]
                index_space = index_space[center - thresholdL:center + thresholdR]
                possible_path.append(search_space)
                possible_index.append(index_space)
                possible_index
                # search and get the best index   
        possible_path = np.array(possible_path).reshape((-1, 4096))
        possible_index = np.array(possible_index).reshape((-1, 1))
        return possible_path, possible_index

def recall_index_pos(i, num_first_part, num_core_part, s, core, e):
    if i < num_first_part:
        return s[0] + i
    elif i < num_first_part + num_core_part:
        return core[0] + i
    else:
        return e[0] + i


def estimate_track_MSPF_SeqSLAM(test_feature, ref_feature, p_id):
    """ Estimate track """
    # caculate image difference matrix
    # startT = time.time()
    m_diff = get_difference_matrix(test_feature, ref_feature)
    # log("Get difference, used time {} for {}".format(time.time()-startT, p_id), mode='g')
    # caculate enhancement matrix
    # startT = time.time()
    # pdb.set_trace()
    m_enhance = get_enhancement_matrix(m_diff)
    # log("Get enhancement, used time {} for {}".format(time.time()-startT, p_id), mode='b')
    # caculate potential matches
    # startT = time.time()
    matches = get_matches(m_enhance)
    # log("Get matches, used time {} for {}".format(time.time()-startT, p_id), mode='y')
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
    features = np.zeros((len(data_path), cfg.img_size * cfg.img_size))
    for count, addr in enumerate(data_path):
        if show:
            print (addr)
        feature = load_image(addr)
        features[count, :] = feature
    return features

def get_feature(img):
    return load_feature(img)

def get_difference_matrix(ref, test):
    """ Get difference matrix """
    diff = Cosine(ref, test)
    diff = np.exp(1 - diff)
    diff = Euclidean(ref, test)
    return diff


def get_enhancement_matrix(diff):
    """ Evalute enhancement matrix """
    ddiff = np.zeros(diff.shape)
    for i in range(diff.shape[0]):
        a = np.max((0, i - cfg.constrastEnhancementR / 2))
        b = np.min((diff.shape[0], i + cfg.constrastEnhancementR / 2 + 1))
        v = diff[a:b, :]
        ddiff[i, :] = (diff[i, :] - np.mean(v, 0)) / np.std(v, 0, ddof=1)
    return ddiff - np.min(np.min(ddiff))


def get_matches(diff):
    """ Get potential matches """
    # Load parameters
    v_ds = cfg.v_ds
    vmax = cfg.vmax
    vmin = cfg.vmin
    # Rwindow = cfg.Rwindow
    matches = np.nan * np.ones((diff.shape[1], 3))
    # parfor?
    for N in range(int(v_ds / 2), diff.shape[1] - int(v_ds / 2)):
        move_min = vmin * v_ds
        move_max = vmax * v_ds
        move = np.arange(int(move_min), int(move_max) + 1)
        v = move.astype(float) / v_ds
        idx_add = np.tile(np.arange(0, v_ds + 1), (len(v), 1))
        idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T)
        # this is where our trajectory starts
        n_start = N + 1 - v_ds / 2
        x = np.tile(np.arange(n_start, n_start + v_ds + 1), (len(v), 1))
        # TODO idx_add and x now equivalent to MATLAB, dh 1 indexing
        score = np.zeros(diff.shape[0])
        # add a line of inf costs so that we penalize running out of data
        diff = np.vstack((diff, np.infty * np.ones((1, diff.shape[1]))))
        y_max = diff.shape[0]
        xx = (x - 1) * y_max
        flatDD = diff.flatten(1)
        for s in range(1, diff.shape[0]):
            y = np.copy(idx_add + s)
            y[y > y_max] = y_max
            idx = (xx + y).astype(int)
            ds = np.sum(flatDD[idx - 1], 1)
            score[s - 1] = np.min(ds)
        min_idx = np.argmin(score)
        min_value = score[min_idx]
        match = [N, min_idx + v_ds / 2, 1. / min_value]
        if match[2] > 1:
            match[2] = 1.0
        matches[N, :] = match
    return matches
