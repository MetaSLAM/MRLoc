import os
import json
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from config import cfg

def save_matches(m_diff, m_enhance, matches, name):
    # save ori matrix
    #m_diff = (m_diff-np.min(m_diff))/(np.max(m_diff)-np.min(m_diff))*255
    #scipy.misc.imsave(os.path.join(cfg.log_dir, name+'_diff.png'), m_diff)
    scipy.misc.imsave(os.path.join(cfg.log_dir, name+'_enhance.png'), m_enhance * 255)
    '''
    ## show matching results
    plt.figure()
    m = matches[:,0]                        # The LARGER the score, the WEAKER the match.
    thresh=0.80                             # you can calculate a precision-recall plot by varying this threshold
    m[matches[:,1]>thresh] = np.nan         # remove the weakest matches
    plt.plot(m,'.')                         # ideally, this would only be the diagonal
    plt.title('Matchings')
    plt.savefig(os.path.join(cfg.log_dir, name+'_match.png'))
    plt.close()
    # save iterative matrixs
    '''
    
def eval_auc(match):
    """Return AUC of current match"""
    match_PR = match[int(cfg.v_ds/2):int(match.shape[0]-cfg.v_ds/2), 1:]
    match_BS = np.array(range(match_PR.shape[0]))+int(int(cfg.v_ds/2))
    match_EE = np.abs(match_PR[:,0] - match_BS)
    match_PR[match_EE<=10, 0] = 1
    match_PR[match_EE> 10, 0] = 0
    match_PR[np.isnan(match_PR)]=0
    fpr, tpr, _ = roc_curve(match_PR[:, 0], match_PR[:, 1])
    roc_auc = auc(fpr, tpr)
    if np.isnan(roc_auc):
        roc_auc = 1.0
    return roc_auc 
    
def save_PR(match):
    # Save match json
    match_PR = match[int(cfg.v_ds/2):int(match.shape[0]-cfg.v_ds/2), :]
    match_BS = np.array(range(match_PR.shape[0]))+int(int(cfg.v_ds/2))
    match_EE = np.abs(match_PR[:,0] - match_BS)
    match_PR[match_EE<=10, 0] = 1
    match_PR[match_EE> 10, 0] = 0
    match_PR[np.isnan(match_PR)]=0
    match_path = os.path.join(cfg.log_dir, 'match.json')
    with open(match_path, 'w') as data_out:
        json.dump(match_PR.tolist(), data_out)

    # Save PR json
    precision, recall, _ = precision_recall_curve(match_PR[:, 0], match_PR[:, 1])
    PR_data = zip(precision, recall) 
    PR_path = os.path.join(cfg.log_dir, 'PR.json')
    with open(PR_path, 'w') as data_out:
        json.dump(PR_data, data_out)

    # Save ROC json
    fpr, tpr, _ = roc_curve(match_PR[:, 0], match_PR[:, 1])
    roc_auc     = auc(fpr, tpr)
    PR_path = os.path.join(cfg.log_dir, 'ROC.json')
    with open(PR_path, 'w') as data_out:
        json.dump(PR_data, data_out)

    # Save PR Curve
    plt.figure()
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Recall/FPR')
    plt.ylabel('Precision/TPR')
    plt.plot(recall, precision, lw=2, color='navy', label='Precision-Recall curve')
    plt.plot(fpr, tpr, lw=2, color='deeppink', label='ROC curve')
    plt.title('PR Curve  (area={0:0.2f})'.format(roc_auc))
    plt.savefig(os.path.join(cfg.log_dir, 'PR.png'))
    plt.close()
    plt.clf()

'''    
    # save match matrix
    match = results.matches
    pairs = match
    print (pairs)
    count = 0
    print (len(pairs))
    print (pairs)
    for id in range(len(pairs)):
        if (np.isnan(pairs[id, 0])):
            continue
        
        print (pairs[id])
        print (ds2.imagePath+'/{0:05}.png'.format(int(id*cfg.frame_skip)))
        print (ds.imagePath+'/{0:05}.png'.format(int(pairs[id, 0])*cfg.frame_skip))
        im1s = cv2.imread(ds2.imagePath+'/{0:05}.png'.format(int(id*cfg.frame_skip)))
        if (abs(int(pairs[id, 0]) - int(id)) <= 20):
            im2s = cv2.imread(ds.imagePath+'/{0:05}.png'.format(int(pairs[id, 0])*cfg.frame_skip))
        else:
            im2s = np.zeros_like(im1s)
            
            h, w = im1s.shape[0], im1s.shape[1]
            
            img = np.zeros((h*1, w*2, 3))
            img[0:h, 0:w, :]     = im1s
            img[0:h, w:2*w, :]   = im2s
                        
            scipy.misc.imsave('{}/{:05d}.png'.format(pair_dir, count), img)
            count +=1
'''            
            
            


