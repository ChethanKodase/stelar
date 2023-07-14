
from functions import extract_image_from_RAS_file_cupd_all
import matplotlib.pyplot as plt
import torch
import numpy as np
datapath = './dataset/lai_ras/'


image_length = 10980

image_width = 10980



import glob
filepaths = glob.glob('./dataset/lai_ras/*.RAS')

filepaths.sort()

#print("filepaths", filepaths)

datapath_filename = filepaths[0]
filename = '32UQV_2002.RAS'

# Run the below separately and not inside any loops
chnl_0_range = 3
chnl_1_range = 2
chnl_2_range = 3
ch_0_max = 255
ch_1_max = 243
ch_2_max = 140
ch0_dist = np.linspace(0, ch_0_max, chnl_0_range).astype(int)
ch1_dist = np.linspace(0, ch_1_max, chnl_1_range).astype(int)
ch2_dist = np.linspace(0, ch_2_max, chnl_2_range).astype(int)


for datapath_filename in filepaths:
    filename = datapath_filename[-14:]
    print("filename", filename)
    img, img_array = extract_image_from_RAS_file_cupd_all(datapath, filename, image_length, image_width)
    img_array = np.array(img_array)

    i_th_day = 0
    for i_th_day in range(len(img_array)):

        for i in range(len(ch0_dist)):
            img_array[i_th_day][:,:,0][abs(img_array[i_th_day][:,:,0] - ch0_dist[i]) <= ch0_dist[1]/2] = ch0_dist[i]

        for i in range(len(ch1_dist)):
            img_array[i_th_day][:,:,1][abs(img_array[i_th_day][:,:,1] - ch1_dist[i]) <= ch1_dist[1]/2] = ch1_dist[i]

        for i in range(len(ch2_dist)):
            img_array[i_th_day][:,:,2][abs(img_array[i_th_day][:,:,2] - ch2_dist[i]) <= ch2_dist[1]/2] = ch2_dist[i]


        P1 = float(271)
        P2 = float(293)
        a=img_array[i_th_day][:,:,0]
        b=img_array[i_th_day][:,:,1]
        c=img_array[i_th_day][:,:,2]

        test_ch = (a*P1 + b)*P2 + c

        test_ch_unqs = np.unique(test_ch)
        #print("test_ch_unqs.shape", test_ch_unqs.shape)

        for i in range(len(test_ch_unqs)):
            test_ch[test_ch == test_ch_unqs[i]] = i
        test_ch = test_ch.astype(int)
        print("test_ch.shape", test_ch.shape)
        #print("test_ch.max()", test_ch.max())
        #print("test_ch.min()", test_ch.min())    
        #print()
        print("day ", i_th_day, " done")

        np.save('./dataset/arbitrary_masks_s/'+datapath_filename[-14:-4]+'_mask_mesr_'+str(i_th_day)+'.npy', test_ch)



