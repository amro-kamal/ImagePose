# from renderer import Renderer
# from strike_utils import *
# from PIL import Image
# import os
# from tabulate import tabulate
# import numpy as np
# jeep=609; brownjeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
# assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508; tablelamp=846;

# # savepath_list = ["data/360/ROLL/bg1/airliner_ROLLPITCH_360/images", "data/360/ROLLPITCH/bg1/ambulance_ROLLPITCH_360/images",
# #                  "data/360/ROLLPITCH/bg1/loader_ROLLPITCH_360/images", "data/360/ROLLPITCH/bg1/brownjeep_ROLLPITCH_360/images"]

# # objectpath_list = [("objects/jeep/jeep.obj", "objects/jeep/jeep.mtl"), ("objects/ambulance/ambulance.obj", "objects/ambulance/ambulance.mtl"),
# #                    ("objects/loader/loader.obj", "objects/loader/loader.mtl"), ("objects/brownjeep/brownjeep.obj", "objects/brownjeep/brownjeep.mtl")]

# # bgpath_list = ["backgrounds/sky.jpg", "backgrounds/sky.jpg", "backgrounds/sky.jpg", "backgrounds/sky.jpg"]
# # pose = 'rollpitch'

# # 1-choose an object
# # 2-choose a pose
# # 3-generate the images for all the bgs

# TRUE_CLASS_list = ['fireengine']
# POSE = 'ROLLPITCH' #'IN_PLANE_ROLL'
# pose = 'rollpitch' #in_plane_roll' 
# bg = 'nobg' #['bg2', 'nobg']

# for a in [1]:
    

#     savepath_list = [f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images"]
#     objectpath_list = [(f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.obj", f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.mtl")]
#     bgpath_list = [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg"] #if os.path.exists(f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg") else [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpeg"] ]

#     if not os.path.exists(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360"):
#        os.mkdir(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360")
#     if not os.path.exists(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images"):
#        os.mkdir(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images")

#     if __name__ == "__main__":
        
#         # Initialize neural network.
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         print('Running on ',device)
#         for obj in range(len(savepath_list)):
#             objpath, mtlpath = objectpath_list[obj]
#             savepath = savepath_list[obj]
#             bgpath = bgpath_list[obj]
#             TRUE_CLASS = TRUE_CLASS_list[obj]
#             print(f'working on object: {objectpath_list[obj][0]} ...........')
#             # Initialize renderer.
#             if bg=='nobg':
#                 renderer = Renderer(objpath, mtlpath)
#             else:
#                 renderer = Renderer(
#                 objpath, mtlpath, bgpath
#                 )
            
#             # print(f'Default renderer parameters: ',renderer.prog["x"].value)
#             #########################    #########################
#             #########################    #########################
#             renderer.prog["x"].value = 0.0
#             renderer.prog["y"].value = 0.0
#             renderer.prog["z"].value = -6
            
#             renderer.prog["amb_int"].value = 0.9#light
#             renderer.prog["dif_int"].value = 1.0
#             DirLight = np.array([1.0, 1.0, 1.0])
#             DirLight /= np.linalg.norm(DirLight)

#             renderer.prog["DirLight"].value = tuple(DirLight)
                
#             for degree in range(0,360,1):
#                 Rot_mat = gen_rotation_matrix(degree, np.pi/10, 0)
#                 image = renderer.render()



# # fireengine:
# #    ROLL: [gen_rotation_matrix(np.pi/10, np.pi/20, degreep), (0, 0.05, -5.5), 0.9] #-6, degreep+np.pi/40
# #    PITCH: [gen_rotation_matrix(np.pi/2.7, 0, degreep+np.pi/45), (0, 0.05, -5.5), 0.9]
# #    YAW:  [gen_rotation_matrix(np.pi/10, degreep+np.pi/20, 0), (0, 0.05, -5.5), 0.9]
# #    IN_PLANE: ROLLphi @ YAWThetaMat @ PITCHThetaMat

                             
#                     R_obj = gen_rotation_matrix(np.pi/10, np.pi/20, np.pi/40) #yaw, pitch, roll # ROLLphi @ YAWThetaMat 

#                     renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
#                     # Render new scene.
#                     image = renderer.render()

#                     image.save(os.path.join(savepath,f'{TRUE_CLASS_list[obj]}_{pose}_{bg}_{i}_{j}.png'))

#             print('images saved to ',savepath_list[obj])

# import moderngl
import moderngl
print(moderngl.__version__)