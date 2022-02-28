#!/usr/bin/env python
'''
File infer_seqslam.py
Function: evaluate seqslam performance
Modified: 2018/9/16
Author: maxtom
'''
import rospy
import roslib
roslib.load_manifest('lcd')
from std_srvs.srv import SetBool

import os
import tensorflow as tf
from config import cfg
from src.seqslam import SeqSlam, MspfSeqSlam, get_feature
from src.util import log

from src.seqslam.mrpfslam import get_difference_matrix, get_enhancement_matrix

import cv2
import sys
import pdb
import numpy as np
from glob import glob
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError

class loop_closure_detection:
    def __init__(self):
        self.image_pub = rospy.Publisher("/raw_image", Image, queue_size=20)
        self.image_sub = rospy.Subscriber("/d435/color/image_raw/compressed", \
                                            CompressedImage, self.callback, queue_size=1)
        self.lcd_trigger = rospy.Service('lcd_request', SetBool, self.lcd_trigger)
        print "Ready to add two ints."
    
        self.bridge = CvBridge()
        self.lcd_detect = MspfSeqSlam()

        self.cur_image = []
        
        # memory frames
        self.frames_memory = []
        self.frames_temp   = []

        # features
        self.features_memory = []
        self.features_temp = []

        self.len_temp   = 0
        self.len_memory = 0

        # Place ID
        self.place_id=0
        self.count = 0

        path_test   = "data/NSHA/test"
        path_ref    = "data/NSHA/refer"

        # NOTE Step1: cache refer frames
        self.test_path = glob(os.path.join(path_test, cfg.synax))
        self.ref_path = glob(os.path.join(path_ref, cfg.synax))
        self.test_path.sort()
        self.ref_path.sort()

        for im_p in self.ref_path:
            img = cv2.imread(im_p)
            feature = get_feature(img)
            self.lcd_memory_cache(img, feature)

        # NOTE Step2: cache new frames
        for im_p in self.test_path:
            img = cv2.imread(im_p)
            feature = get_feature(img)
            self.lcd_temp_cache(img, feature)
            log("Current memory images {}, temp_images {}".format(self.len_memory, self.len_temp), mode='g')
            print (self.len_temp)
            print (cfg.test_size)
            
            if self.len_temp>=cfg.test_size:
                # ANCHOR Step3: Enter LCD if test is full
                log("=================================================================================", mode='b')
                log("Enter Global LCD mode", mode='r')
                log("=================================================================================", mode='b')
                
                D = get_difference_matrix(self.features_temp, self.features_memory)
                DD = get_enhancement_matrix(D)
                # DD   = cv2.resize(DD, (600, 600))
                DD = cv2.normalize(DD, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                DD_name = os.path.join(cfg.log_dir, 'DD_{:04}.png'.format(self.count))
                cv2.imwrite(DD_name, DD*255)

                '''
                print(self.features_memory.shape)
                print(self.features_temp.shape)
                index, DD = self.lcd_detect.ros_global_search(self.features_temp, self.features_memory)

                print(DD)

                index = np.mod(index, self.len_memory)
                img1 = img
                img2 = self.frames_memory[index]
                img3 = cv2.imread(os.path.join(cfg.log_dir, "matrix.jpg"))
                img1 = cv2.resize(img1, (320 * 2, 180 * 2))
                img2 = cv2.resize(img2, (320 * 2, 180 * 2))
                img3 = cv2.resize(img3, (320 * 2, 180 * 2))
                DD   = cv2.resize(DD, (600, 600))
                vis = np.concatenate((img1, img2, img3), axis=1)
                save_name = os.path.join(cfg.log_dir, 'match_{:04}.png'.format(self.count))
                DD_name = os.path.join(cfg.log_dir, 'DD_{:04}.png'.format(self.count))
                print(save_name)
                cv2.imwrite(save_name, vis)
                cv2.imwrite(DD_name, DD*10)
                '''
                self.count += 1
            

    def callback(self,data): 
        np_arr = np.fromstring(data.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image = cv2.resize(cv_image, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_CUBIC)
        self.cur_image = cv_image
        
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)
 
    # Cache memory frames and features
    def lcd_memory_cache(self, img, feature):
        # log("Save frames into the cached memory", mode="r")
        if self.len_memory==0:
            self.frames_memory = img.reshape(1,64,64,3)
            self.features_memory = feature.reshape(1, -1)
        else:
            self.frames_memory = np.concatenate((self.frames_memory, img.reshape(1,64,64,3)), axis=0)
            self.features_memory = np.concatenate((self.features_memory, feature.reshape(1, -1)), axis=0)

        self.len_memory = len(self.frames_memory)

    # Cache temp frames and features
    def lcd_temp_cache(self, img, feature):
        # log("Save frame into local temp frames", mode="g")
        if self.len_temp==0:
            self.frames_temp   = img.reshape(1,64,64,3)
            self.features_temp = feature.reshape(1, -1)
        else:
            if self.len_temp==cfg.test_size:
                self.frames_temp = self.frames_temp[1:]
                self.features_temp = self.features_temp[1:]
            self.frames_temp = np.concatenate((self.frames_temp, img.reshape(1,64,64,3)), axis=0)
            self.features_temp = np.concatenate((self.features_temp, feature.reshape(1, -1)), axis=0)

        self.len_temp = len(self.frames_temp)

    def lcd_memory_reorder(self):
        log("Reorder frames in the cached memory", mode="y")

    def lcd_matching_online(self):
        log("Find the matches in the online mode", mode="y")
        self.lcd_detect.run(self.frames_temp, self.frames_memory)

    #################################################################################
    #                     Loop Closure Detection trigger module                     #
    #################################################################################
    def lcd_trigger(self, req):
        """
        Refer     *000100
        Frames    *000010
                  *000001
                  ******* Temp frames
        NOTE: Cache image into history frames and temp frames, 
        """
        save_name = os.path.join(cfg.log_dir, 'frame_{:04}.png'.format(self.count))
        print(save_name)
        cv2.imwrite(save_name, self.cur_image)
        self.count += 1

        """
        feature = get_feature(self.cur_image)

        # ANCHOR Step1: Cache history frames
        self.lcd_memory_cache(self.cur_image, feature)

        # ANCHOR Step2: Cache temp frames
        self.lcd_temp_cache(self.cur_image, feature)
        log("Current memory images {}, temp_images {}".format(self.len_memory, self.len_temp), mode='g')
        print("")

        if self.len_memory>=cfg.test_size*3:
            # ANCHOR Step3: Enter LCD if test is full
            log("=================================================================================", mode='b')
            log("Enter Global LCD mode", mode='r')
            log("=================================================================================", mode='b')
            print(self.features_memory.shape)
            print(self.features_temp.shape)
            index = self.lcd_detect.ros_global_search(self.features_temp, self.features_memory[:(self.len_memory-cfg.test_size)])

            img1 = self.cur_image
            img2 = self.frames_memory[index]
            img1 = cv2.resize(img1, (320 * 2, 180 * 2))
            img2 = cv2.resize(img2, (320 * 2, 180 * 2))
            vis = np.concatenate((img1, img2), axis=1)
            save_name = os.path.join(cfg.log_dir, 'match_{:04}.png'.format(self.count))
            print(save_name)
            cv2.imwrite(save_name, vis)

            self.count += 1
            # ANCHOR Step4: Online refinement step within SeqSLAM (online mode)
            # self.lcd_matching_online()
            # ANCHOR Step5: LCD refine
            # self.lcd_memory_reorder()

        self.place_id+=1
        """
        return [1, "ok"]

def main(args):
    rospy.init_node('lcd', anonymous=True)

    cfg.method = "MRPF"
    cfg.dataset = "Online"
    cfg.log_dir = os.path.join(cfg.file_path, cfg.dataset, cfg.method)
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    lcd = loop_closure_detection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")        
    cv2.destroyAllWindows()

    # path_test   = "data/NSHA/test1"
    # path_ref    = "data/NSHA/refer_loop"
    # path_test   = "data/NSHA/test"
    # path_ref    = "data/NSHA/refer"
    # mspf_slam = MspfSeqSlam()
    # mspf_slam.run(path_test, path_ref)

if __name__ == '__main__':
    main(sys.argv)