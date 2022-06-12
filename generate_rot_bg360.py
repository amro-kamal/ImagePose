import os
from PIL import Image, ImageDraw
import numpy as np

def rot_image_ramdomly(colorImage, degree, savepath, savename):

    height, width, _  = np.array(colorImage).shape
    rotated_img_arr = np.array(colorImage.rotate(degree, resample=Image.BICUBIC ))
    black_cirlce = Image.new('RGB', [height,width] , color=(123, 116, 103)) #black image
    draw = ImageDraw.Draw(black_cirlce) #white circle
    draw.pieslice([(0,0), (height,width)], 0, 360, fill = 0, outline = "black")

    white_circle = Image.new('L', [height,width] , 0) #black image
    draw = ImageDraw.Draw(white_circle) #white circle
    draw.pieslice([(0,0), (height,width)], 0, 360, fill = 255, outline = "white")
    norm_white_circle = (np.array(white_circle).reshape(height,width,1)/255).astype(np.uint8)

    img_circle = rotated_img_arr*norm_white_circle

    final_img = Image.fromarray(np.array(black_cirlce)+img_circle)

    final_img.save(os.path.join(savepath, f'{savename}'), quality=100, subsampling=0)



def generate_rot_data(colorImage, pose, true_class, savepath):
  for degree in range(360):
    height, width, _  = np.array(colorImage).shape
    rotated_img_arr = np.array(colorImage.rotate(degree, resample=Image.BICUBIC ))
    black_cirlce = Image.new('RGB', [height,width] , color=(123, 116, 103)) #black image
    draw = ImageDraw.Draw(black_cirlce) #white circle
    draw.pieslice([(0,0), (height,width)], 0, 360, fill = 0, outline = "black")

    white_circle = Image.new('L', [height,width] , 0) #black image
    draw = ImageDraw.Draw(white_circle) #white circle
    draw.pieslice([(0,0), (height,width)], 0, 360, fill = 255, outline = "white")
    norm_white_circle = (np.array(white_circle).reshape(height,width,1)/255).astype(np.uint8)

    img_circle = rotated_img_arr*norm_white_circle

    final_img = Image.fromarray(np.array(black_cirlce)+img_circle)

    final_img.save(os.path.join(savepath, f'{true_class}_{pose}_{degree}.png'), quality=100, subsampling=0)


#change the TRUE_CLASS and pose only
TRUE_CLASS = 'loader'
pose='roll'
####################################
savepath = f"data/rot_bg360/ROLL/bg1/{TRUE_CLASS}_ROLL_360/rot_bg_images"
bgImage  = Image.open(f"backgrounds/street_299.jpg")
generate_rot_data(bgImage, pose=pose, true_class=TRUE_CLASS, savepath=savepath)

# savepath = f"data/rot_bg360/ROLL/bg1/{TRUE_CLASS}_ROLL_360/rot_bg_images_lr600"
# bgImage  = Image.open(f"backgrounds/street_lr600.jpg")
# generate_rot_data(bgImage, pose=pose, true_class=TRUE_CLASS, savepath=savepath)


# from renderer import Renderer
# from strike_utils import *
# from PIL import Image
# import os
# from tabulate import tabulate

# jeep=609; brownjeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
# assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508; tablelamp=846;

# objectpath_list = [(f"objects/{TRUE_CLASS}/{TRUE_CLASS}.obj", f"objects/{TRUE_CLASS}/{TRUE_CLASS}.mtl")]

# savepath_list = [f"data/rot_bg360/ROLL/bg1/{TRUE_CLASS}_ROLL_360/rot_bg_images"]
# bgpath_list =   [f"data/rot_bg360/ROLL/bg1/{TRUE_CLASS}_ROLL_360/rot_bg_images/{TRUE_CLASS}_{pose}_{degree}.png" for degree in range(360)]

# # savepath_list = [f"data/rot_bg360/ROLL/bg1/{TRUE_CLASS}_ROLL_360/rot_bg_images_lr600"]
# # bgpath_list =   [f"data/rot_bg360/ROLL/bg1/{TRUE_CLASS}_ROLL_360/rot_bg_images_lr600/{TRUE_CLASS}_{pose}_{degree}.png" for degree in range(360)]
# bgpath = bgpath_list[0]

# if __name__ == "__main__":

#     # Initialize neural network.
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print('Running on ',device)
#     for obj in range(len(savepath_list)):
#         objpath, mtlpath = objectpath_list[obj]
#         savepath = savepath_list[obj]
#         # TRUE_CLASS = TRUE_CLASS_list[obj]

#         print(f'working on object: {objectpath_list[obj][0]} ...........')
#         #########################    #########################
#         # Initialize renderer.
#         renderer = Renderer(
#             objpath, mtlpath, bgpath
#         )
#         renderer.prog["x"].value = 0
#         renderer.prog["y"].value = 0
#         renderer.prog["z"].value = -8
        
#         renderer.prog["amb_int"].value = 0.6 #light
#         renderer.prog["dif_int"].value = 0.9
#         DirLight = np.array([1.0, 1.0, 1.0])
#         DirLight /= np.linalg.norm(DirLight)
#         renderer.prog["DirLight"].value = tuple(DirLight)
#         #########################    #########################

#         for i in range(0,360,360):
#           degreey = i * (np.pi / 180)
#           for j in range(0,360,1):


#             # Alter renderer parameters.
#             degreep = j * (np.pi / 180)
#             R_obj = gen_rotation_matrix(np.pi/2, 0, 0)  #yaw, pitch, roll
#             # if j==0 or j==180:
#             #     renderer.prog["z"].value = -1
#             # else:
#             #     renderer.prog["z"].value = -4
#             renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
#             # Render new scene.
#             image = renderer.render()

#             image.save(os.path.join(savepath,f'{TRUE_CLASS}_{pose}_{j}.png'))
#             if j+1<360:
#                 renderer.set_up_background( background_f= bgpath_list[j+1])
#         print('images saved to ',savepath_list[obj])














