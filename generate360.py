from renderer import Renderer
from strike_utils import *
from PIL import Image
import os
from tabulate import tabulate

jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508

# savepath_list = ["data/360/ROLL/bg1/airliner_ROLLPITCH_360/images", "data/360/ROLLPITCH/bg1/ambulance_ROLLPITCH_360/images",
#                  "data/360/ROLLPITCH/bg1/loader_ROLLPITCH_360/images", "data/360/ROLLPITCH/bg1/brownjeep_ROLLPITCH_360/images"]

# objectpath_list = [("objects/jeep/jeep.obj", "objects/jeep/jeep.mtl"), ("objects/ambulance/ambulance.obj", "objects/ambulance/ambulance.mtl"),
#                    ("objects/loader/loader.obj", "objects/loader/loader.mtl"), ("objects/brownjeep/brownjeep.obj", "objects/brownjeep/brownjeep.mtl")]

# bgpath_list = ["backgrounds/sky.jpg", "backgrounds/sky.jpg", "backgrounds/sky.jpg", "backgrounds/sky.jpg"]
# pose = 'rollpitch'

TRUE_CLASS_list = ['tablelamp']

savepath_list = ["data/360/YAW/bg2/tablelamp_YAW_360/images"]

objectpath_list = [("objects/tablelamp/tablelamp.obj", "objects/tablelamp/tablelamp.mtl")]

bgpath_list = ["backgrounds/wall2_299.png" ]

pose = 'yaw'

if __name__ == "__main__":
    # Initialize neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on ',device)
    for obj in range(len(savepath_list)):
      objpath, mtlpath = objectpath_list[obj]
      savepath = savepath_list[obj]
      bgpath = bgpath_list[obj]
      TRUE_CLASS = TRUE_CLASS_list[obj]
      print(f'working on object: {objectpath_list[obj][0]} ...........')
      # Initialize renderer.
      renderer = Renderer(
          objpath, mtlpath, bgpath
      )
      # print(f'Default renderer parameters: ',renderer.prog["x"].value)
      
      #########################    #########################
      #########################    #########################
      renderer.prog["x"].value = 0
      renderer.prog["y"].value = 0
      renderer.prog["z"].value = -6
      
      renderer.prog["amb_int"].value = 0.3
      renderer.prog["dif_int"].value = 0.9
      DirLight = np.array([1.0, 1.0, 1.0])
      DirLight /= np.linalg.norm(DirLight)
      renderer.prog["DirLight"].value = tuple(DirLight)
      
      
      # for i in range(0,360,360):
          # degreey = i * (np.pi / 180)
      for j in range(0,360,1):
        # Alter renderer parameters.
        degreep = j * (np.pi / 180)
     
        R_obj = gen_rotation_matrix(degreep, np.pi/3.5 , 0)   #yaw, pitch, roll

        renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
        # Render new scene.
        image = renderer.render()

        image.save(os.path.join(savepath,f'{TRUE_CLASS_list[obj]}_{pose}_{j}.png'))


      print('images saved to ',savepath_list[obj])




# from PIL import Image
# for name in os.listdir("data/360/YAW/nobg/assault_rifle_YAW_360/images_lr2000"):

#     image = Image.open(os.path.join("data/360/YAW/nobg/assault_rifle_YAW_360/images_lr2000", name))
#     image = image.resize((299,299),Image.ANTIALIAS)
#     image.save(os.path.join("data/360/YAW/nobg/assault_rifle_YAW_360/images_lr2000resized",'299'+name))