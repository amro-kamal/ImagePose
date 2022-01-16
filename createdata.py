# import os
# import shutil

# '''

# Create dataset of all the objects

# '''
# def copy_rename(src_dir, old_file_name, dst_dir, new_file_name):

#         src_file = os.path.join(src_dir, old_file_name)

#         shutil.copy(src_file,dst_dir)

#         dst_file = os.path.join(dst_dir, old_file_name)

#         new_dst_file_name = os.path.join(dst_dir, new_file_name)

#         os.rename(dst_file, new_dst_file_name)

# obj = 'containership'

# # src_paths = ["data/360/YAW/bg1/"+obj+"_YAW_360/images", "data/360/YAW/bg2/"+obj+"_YAW_360/images", "data/360/YAW/nobg/"+obj+"_YAW_360/images"]
# #             ["data/360/ROLL/bg1/"+obj+"_ROLL_360/images", "data/360/ROLL/bg2/"+obj+"_ROLL_360/images", "data/360/ROLL/nobg/"+obj+"_ROLL_360/images"]
# src_paths =  ["data/360/PITCH/bg1/"+obj+"_PITCH_360/images", "data/360/PITCH/bg2/"+obj+"_PITCH_360/images", "data/360/PITCH/nobg/"+obj+"_PITCH_360/images"]

# axis = ['pitch','pitch','pitch']
# dist_path = "data/combined/"+obj

# if not os.path.isdir(dist_path):
#     os.mkdir(dist_path)

# for i in range(len(src_paths)):
#     src_path = src_paths[i]
#     for img in os.listdir(src_path):
#       if img!= '.DS_Store':
#         try:
#            pose = int(img.split('.')[0].split('_')[-1])
#         except:
#            pose = int(img.split('.')[0].split('_')[-2])
#         if pose<10 or pose>=350:
#             new_name = obj+'_'+axis[i]+str(i%3+1)+'_'+str(pose)+'.'+img.split('.')[1] #<jeep_roll1_10.png>
#             copy_rename(src_path, img, dist_path, new_name)

import matplotlib; matplotlib.use('agg')


import PIL
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-3,3)
plt.plot(x)
fig = plt.gcf()
fig.canvas.draw()

img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())



