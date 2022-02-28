'''
File config.py
Author Maxtom

Function: cache parameters
'''
import tensorflow as tf
param = tf.app.flags

############################
#        SeqSLAM           #
############################
param.DEFINE_string('method',     'MSPF', 'SeqSLAM, MSPF_SeqSLAM')
param.DEFINE_string('dataset',    'Online',  'file name')
param.DEFINE_integer('test_begin', 10000,  'test begin')
param.DEFINE_integer('data_skip',  1,     'skip')
param.DEFINE_integer('init_skip',  1,     'skip')
param.DEFINE_integer('data_len',   400,    'data length')
param.DEFINE_integer('max_depth',  3,      'max_depth')
param.DEFINE_integer('pf_iter',    4,     'max_depth')
param.DEFINE_integer('pf_num',     1000,     'particle number')
param.DEFINE_integer('hog_bin',    64,     'particle number')

## Setting
param.DEFINE_boolean('enResultImages', False, 'output middle images in results dir')
param.DEFINE_integer('test_size', 30, 'Test frames size')
param.DEFINE_integer('test_start', 0, 'Test frames size')

## Particles
param.DEFINE_float('th_particle',  0.8,  'threshold particle')
param.DEFINE_float('tau',  3.0,  'tau')

## SeqSLAM
param.DEFINE_float("v_ds",             10,            "seqslam distance")
param.DEFINE_float("vmin",             0.8,           "min velocity of seqslam")
param.DEFINE_float("vskip",            0.1,           "velocity gap")
param.DEFINE_float("vmax",             1.2,           "max velocity of seqslam")
param.DEFINE_integer("Rwindow",        10,            "rainbow")
param.DEFINE_integer("frame_skip",     2,             "frame skip")    
param.DEFINE_integer("Knn",            5,             "K nearest point")
param.DEFINE_integer("test_base",      0,             "test data base")
param.DEFINE_float("match_distance",   10,            "match threshold for distance")
param.DEFINE_float("match_thres",      20,            "match threshold for GTAV")
param.DEFINE_integer("normalization_sideLength",     8,  "match threshold for GTAV")
param.DEFINE_string('normalization_mode', "global", 'center, global')
param.DEFINE_integer("constrastEnhancementR",      7,   "test data base")

############################
#   environment setting    #
############################
param.DEFINE_boolean('is_crop', True, 'crop the image')
param.DEFINE_string('synax', "*.png", 'png, jpg')
param.DEFINE_integer('img_size', 64, 'png, jpg')

# Directories
param.DEFINE_string('file_path', 'results', 'file path')
param.DEFINE_string('data_dir', 'data', 'data directory')
param.DEFINE_string('log_dir', 'log', 'logs directory')

cfg = param.FLAGS
cfg.data_skip = pow(2, cfg.max_depth) * cfg.init_skip
