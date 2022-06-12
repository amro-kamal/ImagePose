import os 
import random
import shutil
import numpy as np


objects = ['hotdog'] #toaster 1069/3.5, parkingmeter 1050/50, bench=1040/13, 'bicycle'=1030/10, 'motorcycle'=1040/10
data_root = "../../../Documents/"



if not os.path.exists(f"{data_root}/CO3D_subset"):
   os.mkdir(f"{data_root}/CO3D_subset/")
for obj in objects:
    if not os.path.exists(f"{data_root}/CO3D_subset/{obj}"):
      os.mkdir(f"{data_root}/CO3D_subset/{obj}")
    n_images=0
    for folder in os.listdir(f"{data_root}/CO3D objects/{obj}"):
        # if not folder in list(os.listdir(f'{data_root}/old_CO3D_subset/{obj}')):
        #     continue
        if os.path.isdir(f"{data_root}/CO3D objects/{obj}/{folder}"):
            total_folder_images = list(os.listdir(f"{data_root}/CO3D objects/{obj}/{folder}/images"))
            n_total_folder_images = len(total_folder_images)
            
            # if random.random()>0.5:
            #   n_samples = 2 if n_total_folder_images>25  else n_total_folder_images
            # else:
            n_samples = 14 if n_total_folder_images>40  else n_total_folder_images

            ids_20 = random.sample(list(np.arange(0, n_total_folder_images)), n_samples)
            paths_20 = [total_folder_images[i] for i in ids_20]
            #create the place in the CO3D_subset directory
            if not os.path.exists(f"{data_root}/CO3D_subset/{obj}/{folder}"):
                os.mkdir(f"{data_root}/CO3D_subset/{obj}/{folder}")

            if not os.path.exists(f"{data_root}/CO3D_subset/{obj}/{folder}/images"):
                os.mkdir(f"{data_root}/CO3D_subset/{obj}/{folder}/images")

            for img_name in paths_20:
                img_src_path = os.path.join(f"{data_root}/CO3D objects/{obj}/{folder}/images/{img_name}")
                img_dis_path = os.path.join(f"{data_root}/CO3D_subset/{obj}/{folder}/images/{img_name}")
                
                shutil.copyfile( img_src_path , img_dis_path)

            n_images+=n_samples
        # if n_images>1050:
        #     break
    print(f'Finishing from copying {n_images} images from the {obj} folder')


    



for obj in objects:
    n_images=0
    if not os.path.exists(f"{data_root}/CO3D_subset/combined/{obj}"):
      os.mkdir(f"{data_root}/CO3D_subset/combined/{obj}")
      os.mkdir(f"{data_root}/CO3D_subset/combined/{obj}/images")
    for folder in os.listdir(f"{data_root}/CO3D_subset/{obj}"):
        if folder=='.DS_Store':
            continue
        for img_name in os.listdir(f"{data_root}/CO3D_subset/{obj}/{folder}/images"):
            if img_name=='.DS_Store':
                continue
            img_src_path = os.path.join(f"{data_root}/CO3D_subset/{obj}/{folder}/images/{img_name}")
            img_dis_path = os.path.join(f"{data_root}/CO3D_subset/combined/{obj}/images/{folder}_{img_name}")
            shutil.copyfile( img_src_path , img_dis_path)
            n_images+=1




    print(f'Finishing from copying {n_images} images from the {obj} folder')


    





# if not os.path.exists(f"{data_root}/CO3D_subset"):
#    os.mkdir(f"{data_root}/CO3D_subset/")
# for obj in objects:
#     if not os.path.exists(f"{data_root}/CO3D_subset/{obj}"):
#       os.mkdir(f"{data_root}/CO3D_subset/{obj}")
#     n_images=0
#     for folder in os.listdir(f"{data_root}/{obj}"):
#         if not folder in list(os.listdir(f'{data_root}/old_CO3D_subset/{obj}')):
#             continue
#         if os.path.isdir(f"{data_root}/{obj}/{folder}"):
#             total_folder_images = list(os.listdir(f"{data_root}/{obj}/{folder}/images"))
#             n_total_folder_images = len(total_folder_images)

#             n_samples = 25 if n_total_folder_images>25  else n_total_folder_images

#             ids_20 = random.sample(list(np.arange(0, n_total_folder_images)), n_samples)
#             paths_20 = [total_folder_images[i] for i in ids_20]

#             if not os.path.exists(f"{data_root}/CO3D_subset/{obj}/{folder}"):
#                 os.mkdir(f"{data_root}/CO3D_subset/{obj}/{folder}")

#             if not os.path.exists(f"{data_root}/CO3D_subset/{obj}/{folder}/images"):
#                 os.mkdir(f"{data_root}/CO3D_subset/{obj}/{folder}/images")

#             for img_name in paths_20:
#                 img_src_path = os.path.join(f"{data_root}/{obj}/{folder}/images/{img_name}")
#                 img_dis_path = os.path.join(f"{data_root}/CO3D_subset/{obj}/{folder}/images/{img_name}")
                
#                 shutil.copyfile( img_src_path , img_dis_path)


#             n_images+=n_samples

#     print(f'Finishing from copying {n_images} images from the {obj} folder')


    



