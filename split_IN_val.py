import yaml
import os 
from strike_utils import *
from shutil import copyfile, copy
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
mountainbike=671; diningtable = 532; electricfan=545; trafficlight=920; hammerhead=4;  tricycle = 870; unicycle=880;
microwave=651; mixingbowl=659; soupbowl=809; banana=954; hotdog=934; afghanhound=160;
germanshepherd= 235; whiskeyjug=901; siamang=369; swab=840; microwave = 651


####################
labels_path = 'imagenet_val_labels.yml'
obj = microwave
obj_name = 'microwave_651'
######################

with open(labels_path, "r") as ymlfile:
        labels = yaml.load(ymlfile)

data_path = '../../../Downloads/ILSVRC2012_img_val'
images_names = []

labels_file =  open(f'data/IN_val_splited/{obj_name}_labels.txt','w') 

if not os.path.exists(f'data/IN_val_splited/IN_val_{obj_name}'):
    os.mkdir(f'data/IN_val_splited/IN_val_{obj_name}')
    os.mkdir(f'data/IN_val_splited/IN_val_{obj_name}/images')


for img in sort_alphanumerically(os.listdir(data_path)):
    images_names.append(img)

images={}

for i,label in enumerate(labels):
    if obj in label and len(label)==1:
        images[images_names[i]]=labels[i]
for img , label in images.items():
   print('copying: ',(img, label))
   labels_file.write(f'{img}, {[list(imagenet_classes.values())[l] for l in label]} \n')
   copy(os.path.join(data_path,img), f'data/IN_val_splited/IN_val_{obj_name}/images')
print('images saved to: ',f'data/IN_val_splited/IN_val_{obj_name}/images')
#####################
# labels_txt_path = 'ILSVRC2012_validation_ground_truth.txt'
# sea_snake = 65; kite =21
# obj = jeep 
# obj_name = 'jeep'
# #####################

# IN_val_labels_file = open(labels_txt_path, "r")
# labels = IN_val_labels_file.readlines()
# labels = [int(l) for l in labels]

# # print(labels_list[:100])   

# data_path = '../../../Downloads/ILSVRC2012_img_val'
# images_names = []
# obj_labels_file =  open(f'data/new_IN_val_splited/labels/{obj_name}_labels.txt','w') 
    
# if not os.path.exists(f'data/new_IN_val_splited/images/IN_val_{obj_name}'):
#     os.mkdir(f'data/new_IN_val_splited/images/IN_val_{obj_name}')

# paths = os.listdir(data_path)
# paths.sort()
# for img in paths:
#     images_names.append(img)

# images={}


# for i,label in enumerate(labels):
#     if obj == label:
#         images[images_names[i]]=labels[i]
       
# for img , label in images.items():
#    print('copying: ',(img, label))
#    obj_labels_file.write(f'{img}, {list(imagenet_classes.values())[label]} \n')
#    copy(os.path.join(data_path,img), f'data/new_IN_val_splited/images/IN_val_{obj_name}')

# print('images saved to: ',f'data/new_IN_val_splited/images/IN_val_{obj_name}')