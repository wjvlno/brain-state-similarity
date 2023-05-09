# import packages 
import glob
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib import cm,rcParams, cycler
from collections import OrderedDict
cmaps = OrderedDict()
from datetime import datetime
import sys
from statesegmentation import GSBS
import os
import os.path
from os import path
import csv
import pickle
import numpy as np
import pandas as pd
import brainiak.eventseg.event as event
from brainiak.eventseg.event import EventSegment
import nibabel as nib
import brainiak.io as io
import brainiak.utils.fmrisim as fmrisim
from os.path import join as opj
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.matlib
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
plt.rcParams.update({'font.size': 16})
import matplotlib.patches as patches
import copy
import random
from scipy import stats 

# functions (from Baldassano, Heusser)
def generate_event_labels(T, K, length_std):
    event_labels = np.zeros(T, dtype=int)
    start_TR = 0
    for e in range(K - 1):
        length = round(
            ((T - start_TR) / (K - e)) * (1 + length_std * np.random.randn()))
        length = min(max(length, 1), T - start_TR - (K - e))
        event_labels[start_TR:(start_TR + length)] = e
        start_TR = start_TR + length
    event_labels[start_TR:] = K - 1

    return event_labels

def generate_data(V, T, event_labels, event_means, noise_std):
    simul_data = np.empty((V, T))
    for t in range(T):
        simul_data[:, t] = stats.multivariate_normal.rvs(
            event_means[:, event_labels[t]], cov=noise_std, size=1)

    simul_data = stats.zscore(simul_data, axis=1, ddof=1)
    return simul_data

def plot_tt_similarity_matrix(ax, data_matrix, bounds, n_TRs, title_text):
    ax.imshow(np.corrcoef(data_matrix.T), cmap='viridis')
    ax.set_title(title_text)
    ax.set_xlabel('TR')
    ax.set_ylabel('TR')
    # plot the boundaries 
    bounds_aug = np.concatenate(([0],bounds,[n_TRs]))
    for i in range(len(bounds_aug)-1):
        rect = patches.Rectangle(
            (bounds_aug[i],bounds_aug[i]),
            bounds_aug[i+1]-bounds_aug[i],
            bounds_aug[i+1]-bounds_aug[i],
            linewidth=2,edgecolor='w',facecolor='none' # edgecolor = color_var
        )
        ax.add_patch(rect)

def plot_tt_similarity_compare(ax, data_matrix, bounds_1, bounds_2, label_1, label_2, n_TRs, title_text):
    ax.imshow(np.corrcoef(data_matrix.T), cmap='viridis', extent=(0,n_TRs,n_TRs,0), origin='upper') # , extent=(0,445,0,445), origin='upper'

    ax.set_xlabel('TR')
    ax.set_ylabel('TR')
    # plot the boundaries 
    bounds_aug_1 = np.concatenate(([0],bounds_1,[n_TRs]))
    bounds_aug_2 = np.concatenate(([0],bounds_2,[n_TRs]))
    for i in range(len(bounds_aug_1)-1):
        rect = patches.Rectangle(
            (bounds_aug_1[i],bounds_aug_1[i]),
            bounds_aug_1[i+1]-bounds_aug_1[i],
            bounds_aug_1[i+1]-bounds_aug_1[i],
            linewidth=2,edgecolor='w',facecolor='w', alpha = 0.4,
            label=label_1
        )
        ax.add_patch(rect)
    for i in range(len(bounds_aug_2)-1):
        rect = patches.Rectangle(
            (bounds_aug_2[i],bounds_aug_2[i]),
            bounds_aug_2[i+1]-bounds_aug_2[i],
            bounds_aug_2[i+1]-bounds_aug_2[i],
            linewidth=2,edgecolor='orange',facecolor='orange', alpha = 0.4,
            label=label_2
        )   
        ax.add_patch(rect)

    colors = ["w", "orange"]
    texts = [label_1, label_2]
    ps = [ plt.plot([],[],  color=colors[i], marker="o", ms=10, ls="", mec=None,
                label="{:s}".format(texts[i]) )[0]  for i in range(len(texts)) ]
     
    plt.legend(handles=ps, fancybox=True, framealpha=0.5)
    
def reduce_model(m, ev):
    """Reduce a model based on event labels"""
    w = (np.round(ev.segments_[0])==1).astype(bool)
    return np.array([m[wi, :].mean(0) for wi in w.T])
	
    
    
## Segmentation parameters
output_dir = "/scratch/dnasn/outputs/"
input_dir = "/scratch/dnasn/inputs/"
# number of TRs to exclude from WvA
tr_drop = 2
# use subset of N voxels from ROI
roi_subset = 0
roi_size = 300 # n voxels in ROI (roi_size, V)
# z-score within voxel
z_score = 0
# make data distribution positive
pos_dist = 0
# allow NaN segmentation and move on (0 = stop execution if Nan segmentation occurs)
try_seg = 1
# plot and save outputs
plot_results = 1
# save eventseg info (event length, boundaries, etc.)
save_seg = 1
half = 0
split_merge = False


# select inputs from most recent "inputSpec" file in input_dir
list_of_files = glob.glob("{}/*inputSpec*.csv".format(input_dir))
latest_file = max(list_of_files, key=os.path.getctime)

with open(latest_file) as f:
    input_list = [tuple(line) for line in csv.reader(f)]
    
input_list = input_list[1:len(input_list)]

A = pd.read_csv(latest_file)

print(A)

for runit in range(len(input_list)):
    
    sub = input_list[runit][0]
    exam = input_list[runit][1]
    region = input_list[runit][2]
    hemisphere = input_list[runit][3]
    TR_start = int(input_list[runit][4])
    TR_end = int(input_list[runit][5])
    TR_start_c = input_list[runit][4]
    TR_end_c = input_list[runit][5]
    kmax = int(input_list[runit][6])
    T = TR_end - TR_start
    nEvents = np.arange(2,kmax + 1) # np.arange(2,111)
    K_test = nEvents[len(nEvents) - 2]
    sub = str(sub)
    roi_name = region
    
    # handle single digit sub IDs (e.g., "1" -> "001")
    if len(sub) == 1:
        sub = "00{}".format(sub)

    # if multiple exams, concatenate before segmentation
    if len(str(exam)) > 1:
        
        # use first exam in series to preallocate
        exams = np.array(list(str(exam)))
        exam = exams[0]
        
        # labels and such 
        run = "00{}".format(str(int(exam) + 1))
        exam_str = "00{}".format(str(exam))
        file_id = "{}_{}".format(hemisphere, region)

        # non-interpolated image
        no_interp = np.transpose(np.loadtxt("{}/pm{}_ex{}_{}_{}_{}.txt".format(input_dir, sub, str(exam), str(hemisphere), str(region), 'vertex_ts'), delimiter=','))
        img = no_interp
        img = img[~np.all(img==0, axis=1), :]
        # transpose matrix -> TR x voxel
        img = np.transpose(img)
        in_data_concat = img[TR_start:TR_end]

#         # average ts
#         no_interp_avg = np.mean(in_data_concat,0)
#         no_interp_avg_concat = no_interp_avg[TR_start:TR_end]

        # censor frames
        censor_name = "{}/pm{}_bld{}_FDRMS0.2_DVARS50_motion_outliers.txt".format(input_dir, sub, run)
        censor_frames = np.transpose(np.loadtxt(censor_name))
        censor_frames_concat = censor_frames[TR_start:TR_end]

        for exam in exams[1:len(exams)]:
            
            # labels and such
            run = "00{}".format(str(int(exam) + 1))
            exam_str = "00{}".format(str(exam))
            file_id = "{}_{}".format(hemisphere, region)
            
            # non-interpolated image
            no_interp = np.transpose(np.loadtxt("{}/pm{}_ex{}_{}_{}_{}.txt".format(input_dir, sub, str(exam), str(hemisphere), str(region), 'vertex_ts'), delimiter=','))
            
            img = no_interp
            img = img[~np.all(img==0, axis=1), :]
            # transpose matrix -> TR x voxel
            img = np.transpose(img)
            in_data = img[TR_start:TR_end]
            V = in_data.shape[1]
            no_interp_img = no_interp[:,TR_start:TR_end]
            
#             # average ts
#             no_interp_avg = np.mean(no_interp_img,0)
#             no_interp_avg = no_interp_avg[TR_start:TR_end]
            
            # censor frames
            censor_name = "{}/pm{}_bld{}_FDRMS0.2_DVARS50_motion_outliers.txt".format(input_dir, sub, run)
            censor_frames = np.transpose(np.loadtxt(censor_name))
            censor_frames = censor_frames[TR_start:TR_end]
            
            # print("input data:")
            # print("{}/pm{}_ex{}_{}_{}_{}.txt".format(input_dir, sub, str(exam), str(hemisphere), str(region), 'vertex_ts'))
            # print("censor frames:")
            # print(censor_name)
            
            # concatenate within exam
            in_data_concat = np.row_stack([in_data_concat, in_data])
#             no_interp_avg_concat = np.row_stack([no_interp_avg_concat, no_interp_avg])
            censor_frames_concat = np.concatenate([censor_frames_concat, censor_frames])
            
        in_data = in_data_concat
        censor_frames = censor_frames_concat
    else:
        
        # labels and such
        run = "00{}".format(str(int(exam) + 1))
        exam_str = "00{}".format(str(exam))
        file_id = "{}_{}".format(hemisphere, region)
        
        # Run Segmentation
        # data matrices
        # interp = np.transpose(np.loadtxt("{}/pm{}_ex{}_{}_left_lh_parc_avg_vertex_ts.txt".format(input_dir, sub, exam, 'rest_skip5_mc_residc_interp_FDRMS0.2_DVARS50_bp_0.007_1000000_fs6_sm6'), delimiter=','))
        no_interp = np.transpose(np.loadtxt("{}/pm{}_ex{}_{}_{}_{}.txt".format(input_dir, sub, str(exam), str(hemisphere), str(region), 'vertex_ts'), delimiter=','))

        img = no_interp
        img = img[~np.all(img==0, axis=1), :]
        # transpose matrix -> TR x voxel
        img = np.transpose(img)
        in_data = img[TR_start:TR_end]
        V = in_data.shape[1]
#         no_interp_img = no_interp[:,TR_start:TR_end]
        
        # average timeseries
        # no_interp_avg = np.mean(no_interp,0)

        # load data
        censor_name = "{}/pm{}_bld{}_FDRMS0.2_DVARS50_motion_outliers.txt".format(input_dir, sub, run)
        censor_frames = np.transpose(np.loadtxt(censor_name))
        censor_frames = censor_frames[TR_start:TR_end]

    # above here is new as of 8/2 #
    global censored
    censored = np.ndarray.flatten(censor_frames) #[TR_start:TR_end]

    # compute TR x TR correlation matrix
    corrmat = np.corrcoef(in_data)

    # image label for plot and output file 
    img_label = "pm{} exam: {} roi: {}".format(sub, exam, roi_name)
    print(img_label)

    # HMM
    t_distances = np.zeros(len(nEvents)) # new
    t_locals = np.zeros(len(nEvents))
    wd = np.zeros(len(nEvents))
    for i, events in enumerate(nEvents):
        print(f'fitting HMM with {events} events...', end='\r')
        ev = event.EventSegment(n_events=events, n_iter=500, split_merge = split_merge) # , split_merge = False
        ev.fit(in_data, censored) ## EDITS 11/25
        i1, i2 = np.where(np.round(ev.segments_[0])==1)
        if try_seg == 1:
            if i2.size < 1:
                wd[i] = np.nan
                print('failed with %s events...' % (events))
                continue
        w = np.zeros_like(ev.segments_[0])
        w[i1,i2] = 1
        mask = np.dot(w, w.T).astype(bool)

        # mask off within and between event pairs and take average T-distance between all
        within_vec = np.array([])
        across_vec = np.array([])

        # j iterates over events sequentially. instead, iterate over unique events in i2
        unique_events = np.ndarray.tolist(np.unique(i2))

        nUniqueEvents = len(unique_events)

        for j in np.arange(0,nUniqueEvents):
            within_mask = np.zeros(mask.shape, dtype=bool)
            across_mask = np.zeros(mask.shape, dtype=bool)
            if j == nUniqueEvents - 1: # end condition: no across mask
                # within event boundaries

                # lim1 = np.where(i2 == j)[0][0] # jth state starts
                # lim1e = np.where(i2 == j)[0][len(np.where(i2 == j)[0]) - 1] # jth state ends

                lim1 = np.where(i2 == unique_events[j])[0][0] # jth state starts
                lim1e = np.where(i2 == unique_events[j])[0][len(np.where(i2 == unique_events[j])[0]) - 1] # jth state ends

                # make within events mask
                within_mask[:,lim1:lim1e + 1] = 1
                within_mask = np.triu(within_mask, tr_drop)
                within_mask = mask*within_mask
                # populate vectors
#                 within_pairs = np.reshape(corrmat[within_mask], -1, 1)
                within_pairs = np.reshape(corrmat[within_mask], -1)
                within_vec = np.append(within_vec, within_pairs)
            else: 
                # within + between event boundaries
                lim1 = np.where(i2 == unique_events[j])[0][0] # jth state starts
                lim1e = np.where(i2 == unique_events[j])[0][len(np.where(i2 == unique_events[j])[0]) - 1] # jth state ends
                lim2 = np.where(i2 == unique_events[j + 1])[0][0] # j + 1 state begins
                lim2e = np.where(i2 == unique_events[j + 1])[0][len(np.where(i2 == unique_events[j + 1])[0]) - 1] # j + 1 state ends
                # make within events mask
                within_mask[:,lim1:lim1e + 1] = 1
                within_mask = np.triu(within_mask, tr_drop)
                within_mask = mask*within_mask
                # make between events mask 
                across_mask[lim1:lim1e + 1,lim2:lim2e + 1] = 1
                across_mask = np.triu(across_mask, tr_drop)
                # populate vectors
#                 within_pairs = np.reshape(corrmat[within_mask], -1, 1)
                within_pairs = np.reshape(corrmat[within_mask], -1)
#                 across_pairs = np.reshape(corrmat[across_mask], -1, 1)
                across_pairs = np.reshape(corrmat[across_mask], -1)
                within_vec = np.append(within_vec, within_pairs)
                across_vec = np.append(across_vec, across_pairs)
        T_test = stats.ttest_ind(within_vec, across_vec)
        t_distances[i] = T_test.statistic

        # Create mask such that the maximum temporal distance for 
        # within and across correlations is the same
        local_mask = np.zeros(mask.shape, dtype=bool)
        for k in range(mask.shape[0]):
            if ~np.any(np.diag(mask, k)):
                break
            local_mask[np.diag(np.ones(local_mask.shape[0]-k, dtype=bool), k)] = True
#         within_vals = np.reshape(corrmat[mask*local_mask], -1, 1) 
        within_vals = np.reshape(corrmat[mask*local_mask], -1)
#         across_vals = np.reshape(corrmat[~mask*local_mask], -1, 1)
        across_vals = np.reshape(corrmat[~mask*local_mask], -1)
        T_test = stats.ttest_ind(within_vals, across_vals)
        t_locals[i] = T_test.statistic
        wd[i] = wasserstein_distance(within_vals, across_vals)
    maxk_local_tval = nEvents[np.argmax(t_locals)]
    maxk_local_wd = nEvents[np.argmax(wd)]
    maxk_evpair_tdist = nEvents[np.argmax(t_distances)]
    # line1 = 'fitting HMM with %s events...' % (events)
    line2 = 'T-dist optimal K = %s' % (maxk_local_tval)
    line3 = "WD optimal K = %s" % (maxk_local_wd)
    line4 = "Event pair t-dist optimal K = %s" % (maxk_evpair_tdist)
    # print(os.linesep.join([line1, line2, line3, line4]))
    print(os.linesep.join([line2, line3, line4]))

    # re-fit HMM at optimal K
    ev_t = event.EventSegment(maxk_local_tval, n_iter=500, split_merge = split_merge)
    ev_t.fit(in_data, censored)
    roi_events_t = reduce_model(in_data, ev_t)
    pred_seg_t = ev_t.segments_[0]

    ev_wd = event.EventSegment(maxk_local_wd, n_iter=500, split_merge = split_merge)
    ev_wd.fit(in_data, censored) 
    roi_events_wd = reduce_model(in_data, ev_wd)
    pred_seg_wd = ev_wd.segments_[0]

    ev_tep = event.EventSegment(maxk_evpair_tdist, n_iter=500, split_merge = split_merge)
    ev_tep.fit(in_data, censored)
    roi_events_tep = reduce_model(in_data, ev_tep)
    pred_seg_tep = ev_tep.segments_[0]

    # GSBS
    # states = GSBS(x=in_data, kmax=K_test, dmin=tr_drop)
    # states.fit()
    # print("GSBS: optimal K =", states.nstates)

    # plotting   
    # z-score distance metrics
    # tdists_z = states.tdists[2:len(states.tdists)]
    # tdists_z = (tdists_z - np.mean(tdists_z))/np.std(tdists_z)
    t_locals_z = t_locals
    t_locals_z = (t_locals_z - np.mean(t_locals_z))/np.std(t_locals_z)
    wd_z = wd
    wd_z = (wd_z - np.mean(wd_z))/np.std(wd_z)
    tep_z = t_distances ###
    tep_z = (tep_z - np.mean(tep_z))/np.std(tep_z) ###

    fig = plt.figure(constrained_layout=True,figsize=(16,20))
    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax3_1 = fig.add_subplot(gs[1, 0])

    # ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])

    color_var = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    # ax1 = corrmat w/ bounds for HMM w/ WD
    T = in_data.shape[0] # n TRs
    K = maxk_local_wd # n events fit
    bounds = np.where(np.diff(np.argmax(pred_seg_wd, axis=1)))[0]
    title_text = 'HMM w/ WD (WvA): K = %s' % (maxk_local_wd)
    plot_tt_similarity_matrix(ax1, in_data.T, bounds, T, title_text)
    ax1.set_xlabel('time')
    ax1.set_ylabel('time')

    color_var = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
    # ax2 = corrmat w/ bounds for HMM w/ T-dist
    T = in_data.shape[0] # n TRs
    K = maxk_local_tval # n events fit
    bounds = np.where(np.diff(np.argmax(pred_seg_t, axis=1)))[0]
    title_text = 'HMM w/ t-value (WvA): K = %s' % (maxk_local_tval)
    plot_tt_similarity_matrix(ax2, in_data.T, bounds, T, title_text)
    ax2.set_xlabel('time')
    ax2.set_ylabel('time')

    ###
    color_var = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
    # ax3_1 = corrmat w/ bounds for HMM w/ event pair t-dist
    T = in_data.shape[0] # n TRs
    K = maxk_evpair_tdist # n events fit
    bounds = pred_seg_tep.segments_[0] # np.where(np.diff(np.argmax(pred_seg_tep, axis=1)))[0]
    title_text = 'HMM w/ T-distance: K = %s' % (maxk_evpair_tdist)
    plot_tt_similarity_matrix(ax3_1, in_data.T, bounds, T, title_text)
    ax3_1.set_xlabel('time')
    ax3_1.set_ylabel('time')
    ###

    # color_var = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
    # # ax3 = corrmat w/ bounds for GSBS
    # T = in_data.shape[0] # n TRs
    # K = states.nstates # n events fit
    # boundaries = []
    # for i in np.arange(1,K + 1):
    #     arr = np.where(states.states == i)[0]
    #     boundaries.append(arr[0])
    # title_text = 'GSBS w/ T-distance: K = %s' % (states.nstates)
    # plot_tt_similarity_matrix(ax3, in_data.T, boundaries, T, title_text)
    # ax3.set_xlabel('time')
    # ax3.set_ylabel('time')

    # ax4 = t-dist curves
    ax4.plot(nEvents, wd_z, label = "HMM Wass. distance opt K = %s" % (maxk_local_wd), linewidth = 2)
    ax4.plot(nEvents, t_locals_z, label = "HMM T-value WvA opt K = %s" % (maxk_local_tval), linewidth = 2)
    ###
    ax4.plot(nEvents, tep_z, label = "HMM T-distance opt K = %s" % (maxk_evpair_tdist), linewidth = 2)
    ###
    # ax4.plot(nEvents, tdists_z, label = "GSBS T-distance opt K = %s" % (states.nstates), linewidth = 2)
    ax4.set_xlabel('Number of States (K)')
    ax4.set_ylabel('Distance (z-scored)')
    ax4.legend()
    ax4.scatter(maxk_local_wd, wd_z[maxk_local_wd - 2], linewidth = 2, s=100, facecolors='none', edgecolors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    ax4.scatter(maxk_local_tval, t_locals_z[maxk_local_tval - 2], linewidth = 2, s=100, facecolors='none', edgecolors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    ###
    ax4.scatter(maxk_evpair_tdist, tep_z[maxk_evpair_tdist - 2], linewidth = 2, s=100, facecolors='none', edgecolors = plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
    ###
    # ax4.scatter(states.nstates, tdists_z[states.nstates - 2], linewidth = 2, s=100, facecolors='none', edgecolors = plt.rcParams['axes.prop_cycle'].by_key()['color'][3])

    plt.suptitle(img_label, fontsize=32)

    # display plot
    # plt.show()
    
    # save plot
    plt_filename = "s{}_ex{}_{}_b{}_e{}_k{}_plt_bound_corrmat.pdf".format(sub, exam_str, file_id, TR_start_c, TR_end_c, kmax)
    plt.savefig(opj(output_dir, plt_filename), bbox_inches='tight')
    # /home/wvillano/sherlock-topic-model-paper/figures

    # save segmentation outputs
    # save WvA metrics
    data = {'event': nEvents}   
    df = pd.DataFrame(data) 
    df.insert(1, "hmm_wass_z", wd_z, True) 
    df.insert(1, "hmm_tval_z", t_locals_z, True) 
    df.insert(1, "hmm_tdist_z", tep_z, True) 
    # df.insert(1, "gsbs_tdist_z", tdists_z, True) 
    filename = "{}/s{}_ex{}_{}_b{}_e{}_k{}_wva_metrics.csv".format(output_dir, sub, exam_str, file_id, TR_start_c, TR_end_c, kmax)
    df.to_csv(filename)

    # save event indices
    
    # data = {'event': np.arange(1,maxk_evpair_tdist + 1)} 
    # df = pd.DataFrame(data)
    # df.insert(1,'start_TR', np.append(0,bounds))
    filename = "{}/s{}_ex{}_{}_b{}_e{}_k{}_event_indices.csv".format(output_dir, sub, exam_str, file_id, TR_start_c, TR_end_c, kmax)
    np.savetxt(filename, np.append(0,bounds), delimiter=",")
    # df.to_csv(filename)

    # save logLik curves for each TR x event
    df = pd.DataFrame(pred_seg_tep)
    filename = "{}/s{}_ex{}_{}_b{}_e{}_k{}_pred_seg.csv".format(output_dir, sub, exam_str, file_id, TR_start_c, TR_end_c, kmax)
    df.to_csv(filename)

    # save events patterns from segmentation
    df = pd.DataFrame(roi_events_tep.T)
    filename = "{}/s{}_ex{}_{}_b{}_e{}_k{}_event_pat.csv".format(output_dir, sub, exam_str, file_id, TR_start_c, TR_end_c, kmax)
    df.to_csv(filename)

    # save ll_ (as a measure of model fit)
    df = pd.DataFrame(ev_tep.ll_)
    filename = "{}/s{}_ex{}_{}_b{}_e{}_k{}_seg_ll.csv".format(output_dir, sub, exam_str, file_id, TR_start_c, TR_end_c, kmax)
    df.to_csv(filename)

    # save event_var_ (noise)
    data = [['wasserstein', ev_wd.event_var_], ['t_value', ev_t.event_var_], ['t_distance', ev_tep.event_var_]]
    df = pd.DataFrame(data)
    filename = "{}/s{}_ex{}_{}_b{}_e{}_k{}_event_var.csv".format(output_dir, sub, exam_str, file_id, TR_start_c, TR_end_c, kmax)
    df.to_csv(filename)
    
# conda deactivate

# move outputs off scratch to home dir
# mv /scratch/dnasn/outputs/* /home/w.villano/outputs