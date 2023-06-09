
from functions import extract_LAI_from_RAS_file, explore_image
import matplotlib.pyplot as plt
import torch
import numpy as np
datapath = './dataset/lai_ras/'


image_length = 10980

image_width = 10980

select_image = 0

#for i in range(1):
#k=i+1
filename = '32UQV_2001.RAS'
test = extract_LAI_from_RAS_file(datapath, filename, image_length, image_width, select_image)

# convert test to numpy array
test_distri = np.array(test)


'''from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["default"](checkpoint="/home/luser/segment-anything/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
predictor.set_image(test_distri)
masks, _, _ = predictor.predict(<input_prompts>)'''


from segment_anything import segment_anything
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["default"](checkpoint="/home/luser/segment-anything/sam_vit_b_01ec64.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(test_distri)

print(masks)


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


