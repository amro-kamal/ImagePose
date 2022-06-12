# import torchvision
# import matplotlib.pyplot as plt
# import torch
# import cv2
# import glob
# import os
# import re
# import numpy as np

# AIRLINER=404;  BARBERCHAIR = 423; CANNON=471; FIREENGINE=555; FOLDINGCHAIR=559; FORKLIFT=561;
# GARBAGETRUCK = 569; HAMMERHEAD=4; JEEP=609; MOUNTAINBIKE = 671; PARKBENCH=703; ROCKINGCHAIR=765; SHOPPINGCART=791;
# TABLELAMP=846; LAMPSHADE=619; TANK = 847;  TRACTOR = 866; WHEELBARROW = 428; 

# true_class_dict = {'airliner': [AIRLINER], } #'barberchair': [BARBERCHAIR], 'cannon':[CANNON], 'fireengine':[FIREENGINE], 'foldingchair':[FOLDINGCHAIR],
#     # 'forklift':[FORKLIFT], 'garbagetruck':[GARBAGETRUCK], 'hammerhead':[HAMMERHEAD], 'jeep':[JEEP], 'mountainbike':[MOUNTAINBIKE],
#     # 'parkbench':[PARKBENCH], 'rockingchair':[ROCKINGCHAIR], 'shoppingcart':[SHOPPINGCART],
#     # 'tablelamp':[TABLELAMP, LAMPSHADE], 'tank':[TANK], 'tractor':[TRACTOR], 'wheelbarrow':[WHEELBARROW] }


# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
#     return sorted(data, key=alphanum_key)

# POSE='ROLLPITCH'
# bg='bg1'
# model_name = 'Efficientnet_l2_475_noisy_student'
# for obj, c in true_class_dict.items():
#     data_root_path = f"newdata/360/{POSE}/{bg}/{obj}_{POSE}_360/renamed_images_top1"
#     img_array = []

#     for filename in sorted_alphanumeric(glob.glob(os.path.join(data_root_path, f'{model_name}/*.png'))):
#         img = cv2.imread(filename) #229x229x3
#         img_array.append(torch.tensor(img).permute(2,1,0)) #3x229x229

#     batch_tensor = torch.stack(img_array) #torch.randn(*(100, 3, 224, 224))
#     grid_img = torchvision.utils.make_grid(batch_tensor, nrow=10).permute(2, 1, 0)
#     # plt.figure(figsize = (15,15))
#     # plt.imshow(grid_img.permute(2, 1, 0))  #(3, 3012, 3012)==> (3012, 3012, 3)


#     fig, ax = plt.subplots(figsize=(20,20))

#     # ax.set_xticks(np.arange(0, 500*10, 500)+250)
#     # ax.set_xticklabels([36*i for i in range(0,11)], fontsize=12)

#     # ax.set_yticks(np.arange(0, 500*10, 500)+250)
#     # ax.set_yticklabels([36*i for i in range(0,11)],fontsize=12)

#     ax.imshow(grid_img)  #(3, 3012, 3012)==> (3012, 3012, 3)
#     # plt.xlabel(f"Degrees / ROLL - Model: {models_names_map[model_names[m]]}", fontsize=15)
#     plt.ylabel("Degrees / PITCH", fontsize=15)
#     fig.savefig(f'gridplots/{obj}_grid_{bg}_q100.png', bbox_inches='tight', dpi=100)

#     print("DONE", obj)

#     #Y_AXIS ==> ROLL
#     #X_AXIS ==> PITCH



import torchvision
import matplotlib.pyplot as plt
import torch
import cv2
import glob
import os
import re
import numpy as np
from torchvision.io import read_image
from PIL import Image

AIRLINER=404;  BARBERCHAIR = 423; CANNON=471; FIREENGINE=555; FOLDINGCHAIR=559; FORKLIFT=561;
GARBAGETRUCK = 569; HAMMERHEAD=4; JEEP=609; MOUNTAINBIKE = 671; PARKBENCH=703; ROCKINGCHAIR=765; SHOPPINGCART=791;
TABLELAMP=846; LAMPSHADE=619; TANK = 847;  TRACTOR = 866; WHEELBARROW = 428; 

true_class_dict = { 'mountainbike':[MOUNTAINBIKE]}
# {'airliner': [AIRLINER],
#  'barberchair': [BARBERCHAIR], 'cannon':[CANNON], 'fireengine':[FIREENGINE], 'foldingchair':[FOLDINGCHAIR],
#     'forklift':[FORKLIFT], 'garbagetruck':[GARBAGETRUCK], 'hammerhead':[HAMMERHEAD], 'jeep':[JEEP], 'mountainbike':[MOUNTAINBIKE],
#     'parkbench':[PARKBENCH], 'rockingchair':[ROCKINGCHAIR], 'shoppingcart':[SHOPPINGCART],
#     'tablelamp':[TABLELAMP, LAMPSHADE], 'tank':[TANK], 'tractor':[TRACTOR], 'wheelbarrow':[WHEELBARROW] }


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

POSE='ROLLPITCH'
bg='bg1'
model_name =   'BiTM_resnetv2_152x2_448' #'Efficientnet_l2_475_noisy_student'
formal_model_name = 'BiTM-Resnetv2-152x2-448' #'Noisy Student Efficientnet-L2-475'
for obj, c in true_class_dict.items():
    data_root_path = f"newdata/360/{POSE}/{bg}/{obj}_{POSE}_360/renamed_images_top1"
    img_array = []


    for filename in sorted_alphanumeric(glob.glob(os.path.join(data_root_path, f'{model_name}/*.png'))):
        img = read_image(filename) #229x229x3
        
        # img = torchvision.transforms.Resize((400,400))(img) 
        img_array.append(img.permute(0, 2, 1)) #3x229x229

    batch_tensor = torch.stack(img_array) #torch.randn(*(100, 3, 224, 224))clea
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=10)

    fig, ax = plt.subplots(figsize=(18,18))

    ax.set_xticks(np.arange(0, 600*10, 600)+300)
    ax.set_xticklabels([36*i for i in range(0,10)], fontsize=12)

    ax.set_yticks(np.arange(0, 600*10, 600)+300)
    ax.set_yticklabels([36*i for i in range(0,10)],fontsize=12)


    ax.imshow(grid_img.permute(2, 1, 0))  #(3, 3012, 3012)==> (3012, 3012, 3)
    plt.xlabel(f"Degrees / ROLL\nClass: {obj.capitalize()}, Model: {formal_model_name}", fontsize=18)
    plt.ylabel("Degrees / PITCH", fontsize=18)
    fig.savefig(f'gridplots/{model_name}/{obj}_grid_{bg}_{model_name}_q100_500_80_180.pdf', bbox_inches='tight', dpi=80)


    print("DONE", obj)




