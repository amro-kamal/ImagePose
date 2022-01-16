import torch
from CLIP import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torchvision
from torchvision import transforms
import numpy as np
from tabulate import tabulate
import yaml
from PIL import ImageDraw
from strike_utils import imagenet_classes
# import cv2
import numpy as np
import glob
import re
from strike_utils import sort_alphanumerically
import shutil
import matplotlib.pyplot as plt; plt.rcdefaults()
from strike_utils import *
from model_vs_human.modelvshuman.models.pytorch.model_zoo import  vit_large_patch16_224

#TODO
#1- ImagePoseData class

#To delete the .DS_Store file (needed for MacOS ONLY) use the command: find . -name '.#DS_Store' -type f -delete
jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508; lampshade=619
class ImagePoseData(Dataset):

    def __init__(self, root_dir, transform=None):
        self.p = os.listdir(root_dir)
        try:
          self.p.remove('.DS_Store')
        except:
          print("No .DS_Store file")

          
        

        self.transform = transform
        self.root_dir = root_dir
        self.objects = {'jeep':609, 'bench':703, 'ambulance':407, 'traffic_light':920, 'forklift':561, 'umbrella':879}
    def __len__(self):
        return len(self.p)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.p[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, self.p[idx]
#####################
def run_model(model, dataloader, imagenet_classes, savepath,  pose, model_name, model_zoo, true_classes):
    '''
    Function to run vision models

    params: 
        dataloader: bs = len(data)
        imagenet_classes: python list
        correct_class: true label
        savepath: path to save the model_log.txt file and model_result.txt file
        model_zoo: ['torchvision', 'modelvshuman', 'clip']
    return:
        <model_name>_log.txt: a table contains: top1, top5, correct_t_conf, correct_f_conf, wrong_conf
        <model_name>_result.txt: result [image: 'correct' or 'wrong']

        <model_name>_result.yml: result_dict (yaml copy of <model_name>_resut.txt)
        <model_name>_result5.yml: result_dict5 {image_name: [correct/wrong , ids, probs]}
        <model_name>_top5_file.txt: top5 preds tables
    '''
    if not os.path.isdir(savepath):
      os.mkdir(savepath)
    print(f'Running {model_name} on the {imagenet_classes[true_classes[0]]}/{pose} data')
    correct_t_conf, correct_f_conf, wrong_conf=[], [], []
    result_dict = {} #dict to save the either the top1 class is the ture class for each image {(image_name: correct/wrong)}.
    result_file = open (os.path.join(savepath, model_name+"_result.txt"), "w") #file to save the resut dictionary.
    result_dict5 = {}  #dict to save the if the true class is among the top5 preds  {image_name: [correct/wrong , ids, probs]}.
    log = open (os.path.join(savepath, model_name+"_log.txt"), "w") #summary table
    top5_tables = open (os.path.join(savepath, model_name+"_top5_tables.txt"), "a")
    # f = open(os.path.join(savepath, model_name+"_top5_tables.txt"), 'r+')
    top5_tables.truncate(0) # need '0' when using r+
    # f.close()

    correct5, correct=0,0
    print('predicting...')

    for batch, names in dataloader:
    # for img_name in os.listdir(f'newdata/360/ROLL/bg1/tank_ROLL_360/images'):
    #   if img_name == '.DS_Store':
    #         continue
    #     # image = Image.open(os.path.join(data_path, f'rs_jeep{i}.png'))
    #   image = Image.open(os.path.join(f'newdata/360/ROLL/bg1/tank_ROLL_360/images', img_name))
    #   # image = image.resize((224, 224), Image.ANTIALIAS)

    #   # image_tensor = torch.Tensor(np.array(image) / 255.0)
    #   # input_image = image_tensor.permute(2, 0, 1)
    #   image_normalized =  preprocess(image) #preprocess(input_image)
    #   batch = image_normalized[None, :, :, :].to(device)
    #   names = [img_name]
      # batch = preprocess(batch)
      # Calculate features
      with torch.no_grad():
        if model_zoo=='torchvision': #torchvision model
          model.eval()
          output = model(batch.to(device))
          probs = output.softmax(dim=-1)

        elif model_zoo=="modelvshuman": #modelvshuman
          output = model.forward_batch(batch.to(device))
          output = torch.tensor(output)
          probs = output.softmax(dim=-1)

        elif model_zoo=='clip':
          # batch, names=next(iter(dataloader))
          image_input = batch.to(device) #preprocess(batch).unsqueeze(0).to(device)
          text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in imagenet_classes]).to(device) #list(imagenet_classes.values())
          with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
          # Pick the top 5 most similar labels for the image
          image_features /= image_features.norm(dim=-1, keepdim=True)
          text_features /= text_features.norm(dim=-1, keepdim=True)
          similarity = (100.0 * image_features @ text_features.T)
          probs = similarity.softmax(dim=-1) #normalized similarity
      
      print('prediction, Done')
      values, indices = probs.topk(5) #[bs x 5]
      for i in range(values.shape[0]): #For all the <batch_size> images
        #top5
        print(indices[i])
        if len(set(true_classes).intersection(indices[i].tolist())) > 0:    #if true_class in indices[i]: #indices[i].shape = [5]
          correct5+=1
          result_dict5[names[i]] = ['correct', indices[i].tolist(), values[i].tolist()]
        else:
          result_dict5[names[i]] = ['wrong', indices[i].tolist(), values[i].tolist()]
        #top1
        if len(set(true_classes).intersection( [indices[i][0].item()] ) ) > 0: #if true_class==indices[i][0]:
          correct+=1
          result_dict[names[i]] = 'correct'
          correct_t_conf.append(values[i][0].item())
        else:
          result_dict[names[i]] = 'wrong'
          wrong_conf.append(values[i][0].item())
          correct_f_conf.append(probs[i][true_classes[0]].item())
        
      #Create table for top5 predictions
      # for j in range(values.shape[0]): #For all the <batch_size> images
        top5_class_index = ['class index']+[indices[i][j] for j in range(len(indices[i]))]
        # top5_list = ['class']+[(imagenet_classes[indices[j][i]][:30] + '..') if len(imagenet_classes[indices[j][i]]) > 30 else imagenet_classes[indices[j][i]]
        #                                 for i in range(len(indices[j]))]
        top5_list = ['class']+[ re.sub(r'^(.{30}).*$', '\g<1>...', imagenet_classes[indices[i][j]]) for j in range(len(indices[i]))] #indices[j]==[5x1]==>One example
        top5_probs = ['probs']+[values[i][j] for j in range(len(values[i]))]
        top5_correct = ['correct']+[len(set(true_classes).intersection([indices[i][j].item()])) > 0 for j in range(len(indices[i]))]  #['correct']+[indices[i][j]==true_classes for j in range(len(indices[i]))]
        tpo5_table = tabulate([top5_class_index, top5_list, top5_probs, top5_correct], headers=[names[i], '1', '2', '3', "4", "5"])
        top5_tables.write('\n'+tpo5_table+'\n')

    num_images=len(dataloader.sampler)
    print(f'top1 {(correct/num_images):.2f}, top5 {(correct5/num_images):.2f}')
    
    log.write('\n'+'correct_t_conf'+'\n')
    log.write('\n'+'['+','.join([str(elem) for elem in correct_t_conf] )+']'+'\n')
    log.write('\n'+'wrong_conf'+'\n')
    log.write('\n'+'['+','.join([str(elem) for elem in wrong_conf] )+']'+'\n')
    log.write('\n'+'correct_f_conf'+'\n')
    log.write('\n'+'['+','.join([str(elem) for elem in correct_f_conf] )+']'+'\n')


    correct_t_conf = np.mean(correct_t_conf)
    wrong_conf = np.mean(wrong_conf)
    correct_f_conf = np.mean(correct_f_conf)

    table_data =[[pose, num_images, correct, correct5, correct/num_images, correct5/num_images, correct_t_conf, correct_f_conf, wrong_conf, model_name]]
    table = tabulate(table_data, headers=['pose', 'num_images', 'correct', 'correct5', 'top1_acc', 'top5_acc', "correct_t_conf", "correct_f_conf", "wrong_conf", 'model'])
    print (table)
    log.write('\n'+table+'\n')
    log.close()
    result_file.write('\n'+str(result_dict)+'\n')
    result_file.close()
    # print(result_dict)
    #Save result dict and result5 dict to .yml files
    with open(os.path.join(savepath,model_name+'_result.yml'), 'w') as file:
      yaml_file = yaml.dump(result_dict, file)
    with open(os.path.join(savepath,model_name+'_result5.yml'), 'w') as file:
      yaml_file = yaml.dump(result_dict5, file)

    return correct, correct5, np.mean(correct_t_conf), np.mean(correct_f_conf), np.mean(wrong_conf), result_dict, result_dict5
#############################
def rename_img(model_name, images_path,names_file, names_file5, save_path,pose, correct_class):
    '''
    1-Loads the images from images_path and rename them according to the clasifications (correct/wrong) from the names_file.yml file
    then save them to the save_path.
    2-Creates top5 barcharts from the names_file5.yml file.
    '''

    files = glob.glob(os.path.join(save_path, 'renamed_images/*'))
    for f in files:
        os.remove(f)
    files = glob.glob(os.path.join(save_path, 'barplots/*'))
    for f in files:
        os.remove(f)

    k=0
    with open(names_file5, "r") as ymlfile:
        result5 = yaml.load(ymlfile)
    with open(names_file, "r") as ymlfile:
        result = yaml.load(ymlfile)
    for (image_name, classification) in result.items():
        if k%60 == 0:
          print(f'working on image num {k} ...')
        k += 1
        image = Image.open(os.path.join(images_path, image_name))
        class_name = image_name.split('.')[0].split('_')[0]
  
        if pose=='yaw' or pose=='pitch' or pose=='roll':
            p1 = image_name.split('.')[0].split('_')[-1]
            p2=0
            new_name = '_'.join([class_name, model_name, pose, p1, classification+'.'+image_name.split('.')[1]])
        else:
            p1=image_name.split('.')[0].split('_')[-2]
            p2=image_name.split('.')[0].split('_')[-1]
            new_name = '_'.join([class_name, model_name, pose, p1, p2, classification+'.'+image_name.split('.')[1]])
        image.save(os.path.join(save_path, 'renamed_images', new_name))

        indices, values = result5[image_name][1], result5[image_name][2] #result5 = {image_name: [correct/wrong , indices, probs]}
        objects = tuple([imagenet_classes[indices[c]].split(',')[0] for c in range(len(indices))][::-1])
        
        color = ['green' if (indices[c] in correct_class) else 'cyan' for c in range(len(indices))]
        green = False
        for i, c in enumerate(color):
          if c=='green' and green==True:
            color[i] = 'cyan'
          elif c=='green' and green==False:
            green = True
        color = color[::-1]
      
        probs = values[::-1]
        y_pos = np.arange(len(objects))
        plt.barh(y_pos, probs, align='center', alpha=0.5, color=color)
        plt.yticks(y_pos, objects)
        plt.xlabel('probability')
        plt.ylabel('classes')
        plt.title(f'{model_name} top5 probabilities | {pose} degrees: {p1} & {p2}')
        plt.savefig(os.path.join(save_path, 'barplots', new_name),dpi=400)
        # plt.show()
        plt.close()

def create_video(data_root_path, savepath):
    # 1-Add the classification tag to the images
    files = glob.glob(os.path.join(data_root_path, 'joined_renamed_images/*'))
    for f in files:
        os.remove(f)

    print('adding tags to the images')
    correct = Image.open('icons/correct.png')
    correct = correct.resize((70,70))
    wrong = Image.open('icons/wrong.png')
    wrong = wrong.resize((70,70))

    mask_im = Image.new("L", correct.size, 0)
    draw = ImageDraw.Draw(mask_im)
    draw.ellipse((5, 5, 65, 65), fill=255)
    for img_name in os.listdir(os.path.join(data_root_path, 'renamed_images')):
      if img_name=='.DS_Store':
        continue
      back_im = Image.open(os.path.join(data_root_path, 'renamed_images', img_name))
      if img_name.split('.')[0].split('_')[-1] == 'correct':
        back_im.paste(correct, (300, 10),mask_im)
      
      elif img_name.split('.')[0].split('_')[-1] == 'wrong':
        back_im.paste(wrong, (300, 10),mask_im)

      # back_im.save(os.path.join(data_root_path, 'renamed_images', img_name))
      barplot_img = Image.open(os.path.join(data_root_path, 'barplots', img_name))
      barplot_img = barplot_img.resize((600, 600))

      img2 = Image.new("RGB", (600*2, 600), "white")
      img2.paste(back_im, (0, 0))  
      img2.paste(barplot_img, (600, 0))
      img2.save(os.path.join(data_root_path, 'joined_renamed_images', img_name))

      plt.imshow(img2)
      plt.close()

    # 2- Create the video
    print('creating the video')
    img_array = []

    # for filename in sort_alphanumerically(glob.glob(os.path.join(data_root_path,'joined_renamed_images/*.*'))):
    #     img = cv2.imread(filename)
        
    #     height, width, layers = img.shape
    #     size = (width,height)
    #     img_array.append(img)

    
    # out = cv2.VideoWriter(os.path.join(savepath, model_name+'_video.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
    # for i in range(len(img_array)):
    #     out.write(img_array[i])
    # out.release()


    # out = cv2.VideoWriter(os.path.join(savepath, model_name+'_fast_video_top5.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 8, size)
    # for i in range(len(img_array)):
    #     out.write(img_array[i])

    out.release()

if __name__=='__main__':
    jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
    assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508; tablelamp=846; lampshade=619;
    tablelamp=846; tank=847; wheelbarrow = 428; foldingchair=559;
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_names_dict = {'bit':'BiTM_resnetv2_152x2', 'swsl':'ResNeXt101_32x16d_swsl', 'simclr':'simclr_resnet50x1', 'vit':'vit_large_patch16_224', 'clip':'Clip-ViT-B-32'}
    model_name = 'resnet50'
    model_zoo = 'modelvshuman' #'torchvision'
    # clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # model = torchvision.models.resnet50(pretrained=True).to(device) 
    model = vit_large_patch16_224("vit_large_patch16_224")
    pose = ['roll', 'roll', 'roll']

    true_class = [[foldingchair]]*9
    obj = "foldingchair"
    bg = 'bg1'
    POSE = 'ROLL'
    # savepath  = ["data/360/ROLL/bg1/"+obj+"_ROLL_360/model_result/"+model_name, "data/360/ROLL/bg2/"+obj+"_ROLL_360/model_result/"+model_name, "data/360/ROLL/nobg/"+obj+"_ROLL_360/model_result/"+model_name,
    #              "data/360/PITCH/bg1/"+obj+"_PITCH_360/model_result/"+model_name, "data/360/PITCH/bg2/"+obj+"_PITCH_360/model_result/"+model_name, "data/360/PITCH/nobg/"+obj+"_PITCH_360/model_result/"+model_name,
    #              "data/360/YAW/bg1/"+obj+"_YAW_360/model_result/"+model_name, "data/360/YAW/bg2/"+obj+"_YAW_360/model_result/"+model_name, "data/360/YAW/nobg/"+obj+"_YAW_360/model_result/"+model_name]

    # data_root_path = ["data/360/ROLL/bg1/"+obj+"_ROLL_360", "data/360/ROLL/bg2/"+obj+"_ROLL_360", "data/360/ROLL/nobg/"+obj+"_ROLL_360",
    #                   "data/360/PITCH/bg1/"+obj+"_PITCH_360", "data/360/PITCH/bg2/"+obj+"_PITCH_360","data/360/PITCH/nobg/"+obj+"_PITCH_360",
    #                   "data/360/YAW/bg1/"+obj+"_YAW_360", "data/360/YAW/bg2/"+obj+"_YAW_360", "data/360/YAW/nobg/"+obj+"_YAW_360"]
    
    savepath = [f'newdata/360/{POSE} result/{bg}/{obj}/modelresult/{model_name}']
    
    data_root_path =[f'newdata/360/{POSE}/{bg}/{obj}_{POSE}_360'] #[f'newdata/datavalidation/{obj}']
    print('data_root_path ', data_root_path)
    batch_size = 1

    ######################################################################

    for i in range(len(savepath)):
      print(f'working on object number {i+1}/{len(savepath)}')
      data_transform = transforms.Compose([
          transforms.Resize(size=224),
          transforms.CenterCrop(size=(224, 224)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
          )
      ])

      # transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
      # preprocess = transforms.Compose( [ 
      #           # transforms.Scale(224),
      #           transforms.Resize(size=224),
      #           transforms.CenterCrop(size=(224, 224)),
      #           transforms.ToTensor(),
      #           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      #   ]
      #   )
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

      data = ImagePoseData(os.path.join(data_root_path[i], 'images') ,transform=data_transform ) #transforms.ToTensor())
      mydataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    ######################################################################
    # run the model and save the log.txt and result.txt files to the savepath
      correct, correct5, correct_t_conf, correct_f_conf, wrong_conf, result, result5 =  run_model(model, mydataloader,
                                                                              list(imagenet_classes.values()), savepath[i], 
                                                                           pose=pose[i], model_name=model_name, model_zoo=model_zoo, true_classes=true_class[i])
    #####################################################################
      # rename the images with the classifications 
      # print('renaming images')
      
      rename_img(model_name=model_name, images_path=os.path.join(data_root_path[i], 'images'),
                names_file=os.path.join(savepath[i], model_name+'_result.yml'), names_file5=os.path.join(savepath[i], model_name+'_result5.yml'), 
                save_path=data_root_path[i], pose=pose[i], correct_class=true_class[i])

      # # # Create a video from the images
      # create_video(data_root_path[i], savepath[i])







