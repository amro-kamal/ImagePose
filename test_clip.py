import torch
from CLIP import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import numpy as np
from tabulate import tabulate
import yaml
from PIL import ImageDraw
from strike_utils import imagenet_classes
import cv2
import numpy as np
import glob
import re
from strike_utils import sort_alphanumerically

import matplotlib.pyplot as plt; plt.rcdefaults()

#TODO
#1- ImagePoseData class

#To delete the .DS_Store file (ON MacOS ONLY) use the command: find . -name '.#DS_Store' -type f -delete
jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879;
class ImagePoseData(Dataset):

    def __init__(self, root_dir, transform=None):
        self.p = os.listdir(root_dir)
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
def run_model(model, dataloader, imagenet_classes, correct_class, savepath,  pose, model_name, model_zoo='torchvision', true_class=jeep):
    '''
    Function to run vision models
    params: 
        dataloader: bs = len(data)
        imagenet_classes: python list
        correct_class: true label
        savepath: path to save the model_log.txt file and model_result.txt file
        model_zoo: ['torchvision', 'modelvshuman', 'clip']
    return:
        clip_log.txt: top1, top5, correct_t_conf, correct_f_conf, wrong_conf
        clip_result.txt: result [image: 'correct' or 'wrong']
    '''
    correct_t_conf, correct_f_conf, wrong_conf=[], [], []
    result = {} #dict to save the either the top1 class is the ture class for each image (image_name: correct/wrong).
    result_file = open (os.path.join(savepath, model_name+"_result.txt"), "w") #file to save the resut dictionary.
    result5 = {}  #dict to save the if the true class is among the top5 preds  (image_name: correct/wrong).
    log = open (os.path.join(savepath, model_name+"_log.txt"), "w") #summary table
    top5_file = open (os.path.join(savepath, model_name+"_top5.txt"), "a")
    correct5, correct=0,0

    for batch, names in dataloader:
      # Calculate features
      print('predicting...')
      with torch.no_grad():
        if model_zoo=='torchvision': #torchvision model
          output = model(batch.to(device))
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
          
      values, indices = probs.topk(5) #[bs x 5]
      for i in range(values.shape[0]): #For all the <batch_size> images
        #top5
        if true_class in indices[i]: #indices[i].shape = [5]
          correct5+=1
          result5[names[i]] = ['correct', indices[i].tolist(), values[i].tolist()]
        else:
          result5[names[i]] = ['wrong', indices[i].tolist(), values[i].tolist()]
        #top1
        if true_class==indices[i][0]:
          correct+=1
          result[names[i]] = 'correct'
          correct_t_conf.append(values[i][0].item())
        else:
          result[names[i]] = 'wrong'
          wrong_conf.append(values[i][0].item())
          correct_f_conf.append(probs[i][correct_class].item())
        
        #Create table for top5 predictions
        for j in range(values.shape[0]): #For all the <batch_size> images
          top5_class_index = ['class index']+[indices[j][i] for i in range(len(indices[j]))]
          # top5_list = ['class']+[(imagenet_classes[indices[j][i]][:30] + '..') if len(imagenet_classes[indices[j][i]]) > 30 else imagenet_classes[indices[j][i]]
          #                                 for i in range(len(indices[j]))]
          top5_list = ['class']+[ re.sub(r'^(.{30}).*$', '\g<1>...', imagenet_classes[indices[j][i]]) for i in range(len(indices[j]))] #indices[j]==[5x1]==>One example
          top5_probs = ['probs']+[values[j][i] for i in range(len(values[j]))]
          top5_correct = ['correct']+[indices[j][i]==true_class for i in range(len(indices[j]))]
          tpo5_table = tabulate([top5_class_index, top5_list, top5_probs, top5_correct], headers=[names[j], '1', '2', '3', "4", "5"])
          top5_file.write('\n'+tpo5_table+'\n')

    num_images=len(dataloader.sampler)
    print(f'top1 {(correct/num_images):.2f}, top5 {(correct5/num_images):.2f}')
    correct_t_conf = np.mean(correct_t_conf)
    wrong_conf = np.mean(wrong_conf)
    correct_f_conf = np.mean(correct_f_conf)

    table_data =[[pose, num_images, correct, correct5, correct_t_conf, correct_f_conf, wrong_conf, model_name]]
    table = tabulate(table_data, headers=['pose', 'num_images', 'top1', 'top5', "correct_t_conf", "correct_f_conf", "wrong_conf", 'model'])
    print (table)
    log.write('\n'+table+'\n')
    log.close()
    result_file.write('\n'+str(result)+'\n')
    result_file.close()
    print(result)
    #Save result dict and result5 dict to .yml files
    with open(os.path.join(savepath,model_name+'_result.yml'), 'w') as file:
      yaml_file = yaml.dump(result, file)
    with open(os.path.join(savepath,model_name+'_result5.yml'), 'w') as file:
      yaml_file = yaml.dump(result5, file)

    return correct, correct5, np.mean(correct_t_conf), np.mean(correct_f_conf), np.mean(wrong_conf), result, result5

#############################
def rename_img(model_name, images_path,names_file, names_file5, save_path,pose='pose', correct_class=609):
    '''
    Load the images from images_path and rename them according to the clasifications (correct/wrong) from the names_file.yml file
    then save them to the save_path
    '''
    with open(names_file5, "r") as ymlfile:
       result5 = yaml.load(ymlfile)

    with open(names_file, "r") as ymlfile:
        result = yaml.load(ymlfile)
    for (image_name, classification) in result.items():
        image = Image.open(os.path.join(images_path, image_name))
        class_name = image_name.split('.')[0].split('_')[0]
        if pose=='yaw' or pose=='pitch' or pose=='roll':
            p1 = image_name.split('.')[0].split('_')[-1]
            new_name = '_'.join([class_name, model_name, pose, p1, classification+'.png'])
        else:
            p1=image_name.split('.')[0].split('_')[-2]
            p2=image_name.split('.')[0].split('_')[-1]
            new_name = '_'.join([class_name, model_name, pose, p1, p2, classification+'.png'])
        image.save(os.path.join(save_path, 'renamed_images', new_name))

        indices, values = result5[image_name][1], result5[image_name][2]
        objects = tuple([imagenet_classes[indices[c]].split(',')[0] for c in range(len(indices))][::-1])
        color = ['green' if indices[c]==correct_class else 'cyan' for c in range(len(indices))][::-1]
        probs = values[::-1]
        y_pos = np.arange(len(objects))
        plt.barh(y_pos, probs, align='center', alpha=0.5, color=color)
        plt.yticks(y_pos, objects)
        plt.xlabel('probability')
        plt.ylabel('classes')
        plt.title('Topk probabilities')
        plt.savefig(os.path.join(save_path, 'barplots', new_name),dpi=400)
        plt.show()
        plt.close()

def create_video(data_root_path):
      # 1-Add the classification tag to the images

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
      if img_name.split('_')[-1][:-4] == 'correct':
        back_im.paste(correct, (115, 10),mask_im)
      
      elif img_name.split('_')[-1][:-4] == 'wrong':
        back_im.paste(wrong, (115, 10),mask_im)

      # back_im.save(os.path.join(data_root_path, 'renamed_images', img_name))
      barplot_img = Image.open(os.path.join(data_root_path, 'barplots', img_name))
      barplot_img = barplot_img.resize((224, 150))


      img2 = Image.new("RGB", (500, 90), "white")
      img2.paste(back_im, (0, 0))  
      img2.paste(barplot_img, (250, 0))
      img2.save(os.path.join(data_root_path, 'renamed_images', img_name))

      plt.imshow(img2)
      plt.cose()






    # 2- Create the video
    print('creating the video')
    img_array = []

    for filename in sort_alphanumerically(glob.glob(os.path.join(data_root_path,'renamed_images/*.png'))):
        img = cv2.imread(filename)
        
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    
    out = cv2.VideoWriter(os.path.join(savepath,model_name+'_video.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 3, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


    out = cv2.VideoWriter(os.path.join(savepath,model_name+'_fast_video_top5.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
   
if __name__=='__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_names = {'bit':'BiTM_resnetv2_152x2', 'swsl':'ResNeXt101_32x16d_swsl', 'simclr':'simclr_resnet50x1', 'vit':'vit_large_patch16_224'}
    model_name = 'Clip-ViT-B-32'
    model_zoo = 'clip'

    jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879;
    correct_class = jeep

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # model = Model(device,model_name='inceptionv3').to(device) 

    pose = 'rollpitch'

    savepath  = "data/360/ROLLPITCH/bg1/jeep_ROLLPITCH_360/model_result/clip"
    data_root_path = "data/360/ROLLPITCH/bg1/jeep_ROLLPITCH_360"
    batch_size = 360

    ######################################################################
    data_transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),

    ])
    data = ImagePoseData(os.path.join(data_root_path, 'images'),transform=preprocess)
    mydataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    #######################################################################
    # # run the model and save the log.txt and result.txt files to the savepath
    # correct, correct5, correct_t_conf, correct_f_conf, wrong_conf, result, result5 =  run_model(clip_model, mydataloader,list(imagenet_classes.values()), correct_class, savepath, pose=pose, model_name=model_name, model_zoo=model_zoo, true_class=jeep)
    #######################################################################

    # rename the images with the classifications 
    print('renaming images')
    rename_img(model_name=model_name, images_path=os.path.join(data_root_path, 'images'),
              names_file=os.path.join(savepath,model_name+'_result.yml'), names_file5=os.path.join(savepath,model_name+'_result5.yml'), 
              save_path=data_root_path, pose=pose, correct_class=correct_class)

    # Create a video from the images
    create_video(data_root_path)



