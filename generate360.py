from renderer import Renderer
from strike_utils import *
from PIL import Image
import os
from tabulate import tabulate

jeep=609; brownjeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508; tablelamp=846;

# savepath_list = ["data/360/ROLL/bg1/airliner_ROLLPITCH_360/images", "data/360/ROLLPITCH/bg1/ambulance_ROLLPITCH_360/images",
#                  "data/360/ROLLPITCH/bg1/loader_ROLLPITCH_360/images", "data/360/ROLLPITCH/bg1/brownjeep_ROLLPITCH_360/images"]

# objectpath_list = [("objects/jeep/jeep.obj", "objects/jeep/jeep.mtl"), ("objects/ambulance/ambulance.obj", "objects/ambulance/ambulance.mtl"),
#                    ("objects/loader/loader.obj", "objects/loader/loader.mtl"), ("objects/brownjeep/brownjeep.obj", "objects/brownjeep/brownjeep.mtl")]

# bgpath_list = ["backgrounds/sky.jpg", "backgrounds/sky.jpg", "backgrounds/sky.jpg", "backgrounds/sky.jpg"]
# pose = 'rollpitch'

# 1-choose an object
# 2-choose a pose
# 3-generate the images for all the bgs
TRUE_CLASS_list = ['wheelbarrow']
POSE = 'YAW'
pose = 'yaw' 
bg = 'nobg'

savepath_list = [f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images"]
objectpath_list = [(f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.obj", f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.mtl")]
bgpath_list = [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpeg"]
validation_data_path = f"newdata/datavalidation/{TRUE_CLASS_list[0]}/images"

if not os.path.exists(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360"):
  os.mkdir(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360")
if not os.path.exists(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images"):
  os.mkdir(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images")
obj=TRUE_CLASS_list[0]
if not os.path.exists(f'newdata/datavalidation/{obj}'):
    os.mkdir(f'newdata/datavalidation/{obj}')
if not os.path.exists(f'newdata/datavalidation/{obj}/images'):
    os.mkdir(f'newdata/datavalidation/{obj}/images')
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
      if bg=='nobg':
          renderer = Renderer(objpath, mtlpath)
      else:
          renderer = Renderer(
          objpath, mtlpath, bgpath
          )
      
      # print(f'Default renderer parameters: ',renderer.prog["x"].value)
      #########################    #########################
      #########################    #########################
      renderer.prog["x"].value = 0
      renderer.prog["y"].value = 0
      renderer.prog["z"].value = -6.5
      
      renderer.prog["amb_int"].value = 1.0 #light
      renderer.prog["dif_int"].value = 0.6
      DirLight = np.array([1.0, 1.0, 1.0])
      DirLight /= np.linalg.norm(DirLight)
      renderer.prog["DirLight"].value = tuple(DirLight)
      
      for i in range(0,360,360):
          degreey = i * (np.pi / 180)
          for j in range(0,360,1):
            # Alter renderer parameters.
            degreep = j * (np.pi / 180)
            
            R_obj = gen_rotation_matrix(-np.pi/2.2, 0, degreep) #yaw, pitch, roll
            # if j==0 or j==180:
            #     renderer.prog["z"].value = -1
            # else:
            #     renderer.prog["z"].value = -4 
            renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
            # Render new scene.
            image = renderer.render()

            image.save(os.path.join(savepath,f'{TRUE_CLASS_list[obj]}_{pose}_{bg}_{j}.png'))
            if j==0:
              image.show()
            if 0<=j<=10 or 350<=j<=360:
              # if j%2==0:
                image.save(os.path.join(validation_data_path,f'{TRUE_CLASS_list[obj]}_{pose}_{bg}_{j}.png'))

      print('images saved to ',savepath_list[obj])