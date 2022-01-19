import yaml
import os 
from strike_utils import *
from shutil import copyfile, copy, move
# from run_model import run_model, ImagePoseData

'''
Seperate ImageNet classes
'''

jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
assault_rifle=413; whiteshark=2; cannon=471; mug=504; keyboard=508; tablelamp=846; lampshade=619;
containership = 510; cup=968; warplane=895; tigershark=3; tench=0; sea_snake = 65; kite =21; trilobite=69; platypus=103; jellyfish=107 ; Lhasa=204;
tiger_beetle = 300; amphibian=408;  motorscooter  = 670; diningtable=532;  tractor = 866;
golfcart=575; movingvan=675; trailertruck=867; electriclocomotive=547;  foldingchair=559;
freightcar= 565; hourglass = 604; halftrack=586; revolver=763; moped=665; carriage=705; horsecart = 603; sportcar = 817
mountainbike=671; diningtable = 532; electricfan=545; trafficlight=920; hammerhead=4;


####################
POSE = 'PITCH'
pose = 'pitch' 
bg = 'nobg'

if not os.path.exists(f'newdata/360Extra'):
    os.mkdir(f'newdata/360Extra')
if not os.path.exists(f"newdata/360Extra/{POSE}"):
    os.mkdir(f"newdata/360Extra/{POSE}")
if not os.path.exists(f"newdata/360Extra/{POSE}/{bg}"):
    os.mkdir(f"newdata/360Extra/{POSE}/{bg}")

######################

for folder in os.listdir(f"newdata/360/{POSE}/{bg}"):
    # if len(os.listdir(f"newdata/360/{POSE}/{bg}/" + folder)) != 180 and len(os.listdir(f"newdata/360/{POSE}/{bg}/") + folder)) !=360:
        # raise Exception("Images number error: ", folder, len(os.listdir(f"newdata/360/{POSE}/{bg}/") + folder)))
    # print('folder: ',folder)
    if 'DS_Store' in folder:
        os.remove(os.path.join(f"newdata/360/{POSE}/{bg}/" + folder))
        continue

    src_path = f"newdata/360/{POSE}/{bg}/{folder}/images"
    dist_path = f"newdata/360Extra/{POSE}/{bg}/{folder}/images"

    for img_name in os.listdir(src_path):
        if 'DS_Store' in img_name:
            os.remove(os.path.join(src_path, img_name))



    print(folder, len(os.listdir(f"newdata/360/{POSE}/{bg}/{folder}/images")))

    if len(os.listdir(f"newdata/360/{POSE}/{bg}/{folder}/images")) > 200:
        print('okkkkkkkkkkk folder: ',folder)

        src_path = f"newdata/360/{POSE}/{bg}/{folder}/images"
        dist_path = f"newdata/360Extra/{POSE}/{bg}/{folder}/images"

        if not os.path.exists(f"newdata/360Extra/{POSE}/{bg}/{folder}"):
            os.mkdir(f"newdata/360Extra/{POSE}/{bg}/{folder}")
        if not os.path.exists(f"newdata/360Extra/{POSE}/{bg}/{folder}/images"):
            os.mkdir(f"newdata/360Extra/{POSE}/{bg}/{folder}/images")
        
        for img_name in os.listdir(src_path):
            if 'DS_Store' in img_name:
                os.remove(os.path.join(src_path, img_name))

            else:
                angle = int(img_name.split('.')[0].split('_')[-1])
                if angle % 2 != 0:
                    move(os.path.join(src_path, img_name), dist_path)

        

    
# if not os.path.exists(f'newdata/360Extra/datavalidation'):
#     os.mkdir(f'newdata/360Extra/datavalidation')

# ######################

# for folder in os.listdir(f"newdata/datavalidation"):
#     # if len(os.listdir(f"newdata/360/{POSE}/{bg}/" + folder)) != 180 and len(os.listdir(f"newdata/360/{POSE}/{bg}/") + folder)) !=360:
#         # raise Exception("Images number error: ", folder, len(os.listdir(f"newdata/360/{POSE}/{bg}/") + folder)))
#     # print('folder: ',folder)
#     src_path = f"newdata/datavalidation/{folder}/images"
#     dist_path = f"newdata/360Extra/datavalidation/{folder}/images"

#     if 'DS_Store' in folder:
#         os.remove(os.path.join(f"newdata/datavalidation/" + folder))
#         continue

            
#     for img_name in os.listdir(src_path):
#         if 'DS_Store' in img_name:
#             os.remove(os.path.join(src_path, img_name))

#     print(folder, len(os.listdir(f"newdata/datavalidation/{folder}/images")))
#     if len(os.listdir(f"newdata/datavalidation/{folder}/images")) > 100:
#         print('okkkkkkkkkkk folder: ',folder)



#         if not os.path.exists(f"newdata/360Extra/datavalidation/{folder}"):
#             os.mkdir(f"newdata/360Extra/datavalidation/{folder}")
#         if not os.path.exists(f"newdata/360Extra/datavalidation/{folder}/images"):
#             os.mkdir(f"newdata/360Extra/datavalidation/{folder}/images")
        
#         for img_name in os.listdir(src_path):
#             if 'DS_Store' in img_name:
#                 os.remove(os.path.join(src_path, img_name))

#             else:
#                 angle = int(img_name.split('.')[0].split('_')[-1])
#                 if angle % 2 != 0:
#                     move(os.path.join(src_path, img_name), dist_path)

        

    

    



