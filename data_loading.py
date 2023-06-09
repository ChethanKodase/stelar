
from functions import extract_LAI_from_RAS_file, explore_image, extract_all_LAI_from_RAS_file
import matplotlib.pyplot as plt
import torch
import numpy as np
datapath = './dataset/lai_ras/'


image_length = 10980

image_width = 10980

select_image = 0

#for i in range(1):
#k=i+1
year = 20
month=1


import glob
filepaths = glob.glob('./dataset/processed_lai_npy/*.npy')

filepaths.sort()

print("filepaths", filepaths)
print()
print("len(filepaths)", len(filepaths))
#datapath_filename = filepaths[0]
#filename = '32UQV_2002.RAS'
#print("datapath_filename", datapath_filename)
for datapath_filename in filepaths:
    print("datapath_filename", datapath_filename)
    #test = extract_all_LAI_from_RAS_file(datapath_filename, image_length, image_width)
    test = np.load(datapath_filename)
    print(test.shape)
    '''for i in range(len(test)):
        print("test no : "+str(i)+" : ", test[i].shape)
        #save numpy array as .npy file
        print("datapath_filename[-14:-4]+'_measure_'+str(i) : ", datapath_filename[-14:-4]+'_measure_'+str(i))
        np.save('./dataset/processed_lai_npy/'+datapath_filename[-14:-4]+'_measure_'+str(i)+'.npy', test[i])'''


'''print("len(test)", len(test))
print('test[0].shape', test[0].shape)
print('test[1].shape', test[1].shape)'''

#test = test[0]

test[test<0] = 0

# convert test to numpy array
test_distri = np.array(test)

print('test_distri.max()', test_distri.max())
print('test_distri.max()', test_distri.min())
# plot the histogram of the values in the numpy array test
plt.hist(test_distri.flatten(), bins=100)
plt.savefig('./view_check/distri.png')
plt.close()



print('test.shape', test.shape)

plt.imshow(test)
plt.colorbar()
plt.savefig('./view_check/test1.png')
plt.close()




'''filename = '32UQV_2001.RAS'
test = extract_LAI_from_RAS_file(datapath, filename, image_length, image_width, select_image)
test[test<0] = 0

print('test.shape', test.shape)
plt.imshow(test)
plt.colorbar()
plt.savefig('/media/chethan/New Volume/1 A FI CODE/stelar/view_check/test'+str(k)+'.png')
plt.close()'''

#filepath = '/home/luser/Stelar project/dataset/lai_ras/32UQV_2001.RAS'

#explore_image(filepath)