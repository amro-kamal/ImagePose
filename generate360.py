from renderer import Renderer
from strike_utils import *
from PIL import Image
import os
from tabulate import tabulate
import numpy as np
jeep=609; brownjeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508; tablelamp=846;



# rockingchair:
#    ROLL: [ gen_rotation_matrix(+np.pi/10, 0, degreep) , (0, 0, 0.75), -0.75 ]

TRUE_CLASS_list = ['rockingchair']
POSE = 'IN_PLANE_ROLL'
pose = 'in_plane_roll' 
bg = 'bg2'

YAWTheta = np.pi/10
ROLLTheta = -np.pi/10
PITCHTheta = -np.pi/10


savepath_list = [f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images"]
objectpath_list = [(f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.obj", f"objects/{TRUE_CLASS_list[0]}/{TRUE_CLASS_list[0]}.mtl")]
bgpath_list = [f"backgrounds/{TRUE_CLASS_list[0]}_{bg}_500.jpg"]
validation_data_path = f"newdata/datavalidation/IN_PLANE_ROLL/{TRUE_CLASS_list[0]}/images"

if not os.path.exists(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360"):
  os.mkdir(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360")
if not os.path.exists(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images"):
  os.mkdir(f"newdata/360/{POSE}/{bg}/{TRUE_CLASS_list[0]}_{POSE}_360/images")
obj=TRUE_CLASS_list[0]
if not os.path.exists(f'newdata/datavalidation/IN_PLANE_ROLL/{obj}'):
    os.mkdir(f'newdata/datavalidation/IN_PLANE_ROLL/{obj}')
if not os.path.exists(f'newdata/datavalidation/IN_PLANE_ROLL/{obj}/images'):
    os.mkdir(f'newdata/datavalidation/IN_PLANE_ROLL/{obj}/images')
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
      renderer.prog["x"].value = 0.0
      renderer.prog["y"].value = 0.0
      renderer.prog["z"].value = -7.5
      
      renderer.prog["amb_int"].value = 0.75#light
      renderer.prog["dif_int"].value = 1.0
      DirLight = np.array([.3, 1.0, 1.0])
      DirLight /= np.linalg.norm(DirLight)

      renderer.prog["DirLight"].value = tuple(DirLight)
      
      YAWThetaMat = np.array( [[ np.cos(YAWTheta) , 0.    ,       np.sin(YAWTheta) ],
                               [ 0.               , 1.    ,       0.               ],
                               [-np.sin(YAWTheta) , 0.    ,       np.cos(YAWTheta) ]])

                                     
      YAW_90 = np.array(       [[ np.cos(-90 * (np.pi / 180) ) , 0.    ,       np.sin(-90 * (np.pi / 180) )   ],
                               [ 0.                           , 1.         ,       0.               ],
                               [-np.sin(-90 * (np.pi / 180) ) , 0.    ,       np.cos(-90 * (np.pi / 180) )   ]])


      ROLLThetaMat = np.array( [[ np.cos(ROLLTheta) ,-np.sin(ROLLTheta)       , 0  ],
                                [ np.sin(ROLLTheta) , np.cos(ROLLTheta)       , 0  ],
                                [ 0.                , 0.                      , 1. ]])
 

      PITCHThetaMat = np.array( [[  1,                  0     ,                0     ],
                                 [  0,     np.cos(PITCHTheta) , -np.sin(PITCHTheta)  ],
                                 [  0,     np.sin(PITCHTheta) ,  np.cos(PITCHTheta)  ]])
      
      for i in range(0,360,360):
          degreey = i * (np.pi / 180) 
          for j in range(0,360,1):
            # Alter renderer parameters.
            degreep = j * (np.pi / 180) 

            ROLLphi = np.array( [[ np.cos(degreep) ,-np.sin(degreep)         , 0  ],
                                 [ np.sin(degreep) , np.cos(degreep)         , 0  ],
                                 [ 0.              , 0.                      , 1. ]])


            PITCHphi = np.array([[ 1,                 0    ,         0         ],
                                 [  0,     np.cos(degreep) , -np.sin(degreep)  ],
                                 [  0,     np.sin(degreep) ,  np.cos(degreep)  ]])

            R_obj = ROLLphi @ YAWThetaMat #yaw, pitch, roll

            renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
            # Render new scene.
            image = renderer.render()

            image.save(os.path.join(savepath,f'{TRUE_CLASS_list[obj]}_{pose}_{bg}_{j}.png'))
            # if j==0:
            #   image.show()
            if 0<=j<=10 or 350<=j<=360:
              # if j%2==0:
                image.save(os.path.join(validation_data_path,f'{TRUE_CLASS_list[obj]}_{pose}_{bg}_{j}.png'))

      print('images saved to ',savepath_list[obj])