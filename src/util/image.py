import numpy as np
import tensorflow as tf
import cv2
from config import cfg

def load_image(addr):
    """ read an image and resize to (64, 64) """
    # load image
    img = cv2.imread(addr)
    # image resize
    img = cv2.resize(img, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_CUBIC)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image normalization
    img = patchNormalize(img)
    #img = img.flatten()
    return img

def load_feature(img):
    """ read an image and resize to (64, 64) """
    # image resize
    img = cv2.resize(img, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_CUBIC)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image normalization
    img = patchNormalize(img)
    #img = img.flatten()
    return img

def extract_image_feature(addr):
    """ read an image and resize to (64, 64) """
    # load image
    img = cv2.imread(addr)
    # image resize
    img = cv2.resize(img, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_CUBIC)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image normalization
    #hog = cv2.HOGDescriptor()
    #feature = hog.compute(img)
    return hog(img).flatten()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def color_normal(img):
    img[:,:,0]=cv2.equalizeHist(img[:,:,0])
    img[:,:,1]=cv2.equalizeHist(img[:,:,1])
    return img

def patchNormalize(img):
    """ Patch Normalization with """
    s = cfg.normalization_sideLength
    n = range(0, img.shape[0]+2, s)
    m = range(0, img.shape[1]+2, s)   
    for i in range(len(n)-1):
        for j in range(len(m)-1):
            p = img[n[i]:n[i+1], m[j]:m[j+1]]
            pp=np.copy(p.flatten(1))
            #print (np.std(pp, ddof=1))
            if cfg.normalization_mode == "center":
                pp=pp.astype(float)
                img[n[i]:n[i+1], m[j]:m[j+1]] = 127 + np.reshape(np.round((pp-np.mean(pp))/np.std(pp, ddof=1)), (s, s))
            elif cfg.normalization_mode == "global":
                f = 255.0/np.max((1, np.max(pp) - np.min(pp)))
                img[n[i]:n[i+1], m[j]:m[j+1]] = np.round(f * (p-np.min(pp)))
    img = img.flatten(0)
    return img

def hog(img):
    bin_n = cfg.hog_bin
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
