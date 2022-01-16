import torch
from PIL import Image, ImageFont, ImageDraw
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
import shutil
import time
#TODO
#1- ImagePoseData class

#To delete the .DS_Store file (needed for MacOS ONLY) use the command: find . -name '.#DS_Store' -type f -delete
jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
assault_rifle=413; white_shark=2; cannon=471; mug=504; keyboard=508

def clear_folder(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

#############################
def rename_img(model_name, images_path,names_file, names_file5, save_path,pose='pose', correct_class=609):
    '''
    1-Loads the images from images_path and rename them according to the clasifications (correct/wrong) from the names_file.yml file
    then save them to the save_path.
    2-Creates top5 barcharts from the names_file5.yml file.

    renamed_images_top1
    renamed_images_top5
    barplots_top1
    barplots_top5


    '''
    print('images will be saved to,',os.path.join(save_path, 'renamed_images_top1'))
    # if os.path.isdir(os.path.join(save_path, 'renamed_images_top1')):
    #     shutil.rmtree(os.path.join(save_path, 'renamed_images_top1'))
    #     os.mkdir(os.path.join(save_path, 'renamed_images_top1'))
    # else:
    #     os.mkdir(os.path.join(save_path, 'renamed_images_top1'))
    # if os.path.isdir(os.path.join(save_path, 'renamed_images_top5')):
    #     shutil.rmtree(os.path.join(save_path, 'renamed_images_top5'))
    #     os.mkdir(os.path.join(save_path, 'renamed_images_top5'))
    # else:
    #     os.mkdir(os.path.join(save_path, 'renamed_images_top5'))

    # if not os.path.isdir(os.path.join(save_path, 'barplots_top1')):
    #     os.mkdir(os.path.join(save_path, 'barplots_top1'))
    # if not os.path.isdir(os.path.join(save_path, 'barplots_top5')):
    #     os.mkdir(os.path.join(save_path, 'barplots_top5'))

    clear_folder(os.path.join(save_path, 'renamed_images_top1'))
    clear_folder(os.path.join(save_path, 'renamed_images_top5'))
    clear_folder(os.path.join(save_path, 'barplots_top1'))
    clear_folder(os.path.join(save_path, 'barplots_top5'))

    k=0
    with open(names_file5, "r") as ymlfile:
        result5 = yaml.load(ymlfile)
    with open(names_file, "r") as ymlfile:
        result = yaml.load(ymlfile)
    start1 = time.time()
    for (image_name, classification) in result.items():
        pred_class = imagenet_classes[result5[image_name][1][0]].split(',')[0]
        if k%100==0:
          print(f'image num {k} ...')
        k+=1
        image = Image.open(os.path.join(images_path, image_name))
        class_name = image_name.split('.')[0].split('_')[0]
        if pose=='yaw' or pose=='pitch' or pose=='roll':
            p1 = image_name.split('.')[0].split('_')[-1]
            p2=0
            new_name = '_'.join([class_name, model_name, pose, p1, pred_class, classification+'.'+image_name.split('.')[1]])
        else:
            p1=image_name.split('.')[0].split('_')[-2]
            p2=image_name.split('.')[0].split('_')[-1]
            new_name = '_'.join([class_name, model_name, pose, p1, p2, pred_class, classification+'.'+image_name.split('.')[1]])

        image.save(os.path.join(save_path, 'renamed_images_top1', new_name))

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
        plots = plt.barh(y_pos, probs, align='center', alpha=0.5, color=color)

        plt.yticks(y_pos, objects)
        plt.xlabel('probability')
        plt.ylabel('classes')
        plt.title(f'Top5 probabilities | {pose} degrees: {p1} & {p2}')
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5,alpha=0.8)
        for bar in plots.patches:
            plt.text(bar.get_width()+0.01, bar.get_y()+0.4,
                    str(round((bar.get_width()), 2))+'%',
                    fontsize=9, fontweight='bold',
                    color='grey')
        plt.savefig(os.path.join(save_path, 'barplots_top1', new_name), bbox_inches='tight', dpi=500)
        # plt.show()
        plt.close()

    start2 = time.time()
    print(f'renaming images time {start2-start1}')

    print('images saved at: ', os.path.join(save_path, 'renamed_images_top1'))
    #top5
    k=0
    for (image_name, classification) in result5.items():
        if k%100==0:
          print(f'image num {k} ...')
        k+=1
        image = Image.open(os.path.join(images_path, image_name))
        class_name = image_name.split('.')[0].split('_')[0]
        pred_class = imagenet_classes[result5[image_name][1][0]].split(',')[0]

        if pose=='yaw' or pose=='pitch' or pose=='roll':
            p1 = image_name.split('.')[0].split('_')[-1]
            p2=0
            new_name = '_'.join([class_name, model_name, pose, p1, pred_class, classification[0]+'.'+image_name.split('.')[1]])
        else:
            p1=image_name.split('.')[0].split('_')[-2]
            p2=image_name.split('.')[0].split('_')[-1]
            new_name = '_'.join([class_name, model_name, pose, p1, p2, pred_class, classification[0]+'.'+image_name.split('.')[1]])
        image.save(os.path.join(save_path, 'renamed_images_top5', new_name))

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
        plots = plt.barh(y_pos, probs, align='center', alpha=0.5, color=color)

        plt.yticks(y_pos, objects)
        plt.xlabel('probability')
        plt.ylabel('classes')
        plt.title(f'Top5 probabilities | {pose} degrees: {p1} & {p2}')
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5,alpha=0.8)
        for bar in plots.patches:
                plt.text(bar.get_width()+0.01, bar.get_y()+0.4,
                        str(round((bar.get_width()), 2))+'%',
                        fontsize=9, fontweight='bold',
                        color='grey')
        plt.savefig(os.path.join(save_path, 'barplots_top5', new_name), bbox_inches='tight', dpi=500)
        # plt.show()
        plt.close()

    print('images saved at: ', os.path.join(save_path, 'renamed_images_top5'))


def create_video(data_root_path, savepath, model_name, top='top1'):
    # 1-Add the classification tag to the images
    # font = ImageFont.truetype('font.ttf', 200)

    # if not os.path.isdir(os.path.join(data_root_path, f'{top}_joined_renamed_images')):
    #     os.mkdir(os.path.join(data_root_path, f'{top}_joined_renamed_images'))
    clear_folder(os.path.join(data_root_path, f'{top}_joined_renamed_images'))
    
    print('adding tags to the images')
    correct = Image.open('icons/correct.png')
    correct = correct.resize((70,70))
    wrong = Image.open('icons/wrong.png')
    wrong = wrong.resize((70,70))

    start1 = time.time()

    mask_im = Image.new("L", correct.size, 0)
    draw = ImageDraw.Draw(mask_im)
    draw.ellipse((5, 5, 65, 65), fill=255)
    for img_name in os.listdir(os.path.join(data_root_path, f'renamed_images_{top}')):
      if img_name=='.DS_Store':
        continue
      back_im = Image.open(os.path.join(data_root_path, f'renamed_images_{top}', img_name)).resize((600,600))

      if img_name.split('.')[0].split('_')[-1] == 'correct':
        back_im.paste(correct, (280, 10),mask_im)
        # draw = ImageDraw.Draw(back_im)
        # draw.text((15,15), img_name.split('.')[0].split('_')[-2], (237, 230, 211))
      elif img_name.split('.')[0].split('_')[-1] == 'wrong':
        back_im.paste(wrong, (280, 10),mask_im)
        # draw = ImageDraw.Draw(back_im)
        # draw.text((15,15), img_name.split('.')[0].split('_')[-2], (237, 230, 211))

      # back_im.save(os.path.join(data_root_path, 'renamed_images', img_name))
      barplot_img = Image.open(os.path.join(data_root_path, f'barplots_{top}', img_name))
      barplot_img = barplot_img.resize((600, 600))

      img2 = Image.new("RGB", (600*2, 600), "white")
      img2.paste(back_im, (0, 0))  
      img2.paste(barplot_img, (600, 0))
      img2.save(os.path.join(data_root_path, f'{top}_joined_renamed_images', img_name))

      # plt.imshow(img2)
      plt.close()

    start2 = time.time()
    print(f'adding tags time {start2-start1}')
    # 2- Create the video
    print('creating the video')
    img_array = []

    for filename in sort_alphanumerically(glob.glob(os.path.join(data_root_path,f'{top}_joined_renamed_images/*.png'))):
        img = cv2.imread(filename)
        
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    start3= time.time()
    print(f'generating the video time: {start3-start2}')
    print('saving  the video')
    out = cv2.VideoWriter(os.path.join(savepath, model_name, model_name+f'_video_{top}.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    out = cv2.VideoWriter(os.path.join(savepath, model_name, model_name+f'_fast_video_{top}.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 8, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
   
if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # models_names = {'bit':'BiTM-resnetv2-152x2', 'swsl':'ResNeXt101-32x16d-swsl', 'simclr':'simclr-resnet50x1', 'vit':'vit-large-patch16-224'}
    # model_names = ['BiTM-resnetv2-152x2', 'Clip-ViT-B-32','resnet50', 'resnet152','ResNeXt101-32x16d-swsl', 'simclr-resnet50x1', 'vit-large-patch16-224']
    
    model_names = ['BiTM-resnetv2-152x2', 'Clip-ViT-B-32', 'ResNeXt101-32x16d-swsl']
    jeep=609; bench=703; ambulance=407; traffic_light=920; forklift=561; umbrella=879; airliner=404; 
    assault_rifle=413; whiteshark=2; cannon=471; mug=504; keyboard=508; containership=510; tablelamp=846; lampshade=619;

    correct_class = [[tablelamp, lampshade],[tablelamp, lampshade],[tablelamp, lampshade]]
    
    pose = ['roll','roll','roll']

    savepath  = [ "data/360/ROLL result/result/bg1/whiteshark_ROLL_360/", "data/360/ROLL result/result/nobg/whiteshark_ROLL_360/"]
    data_root_path = [ "data/360/ROLL/bg1/whiteshark_ROLL_360", "data/360/ROLL/nobg/whiteshark_ROLL_360"]
    
        ######################################################################
    for m in range(len(model_names)):
        print('current model : ',model_names[m])
        for i in range(len(savepath)):
          start = time.time()

          print(f'working on object number {i+1}/{len(savepath)}')

          # rename the images with the classifications 
          print('renaming images')
          if os.path.isdir(os.path.join(data_root_path[i], 'images_lr600')):
              data_path = os.path.join(data_root_path[i], 'images_lr600')
          else:
              data_path = os.path.join(data_root_path[i], 'images')

          rename_img(model_name=model_names[m], images_path=data_path,
                    names_file=os.path.join(savepath[i], model_names[m],model_names[m]+'_result.yml'), names_file5=os.path.join(savepath[i],model_names[m], model_names[m]+'_result5.yml'), 
                    save_path=data_root_path[i], pose=pose[i], correct_class=correct_class[i])

          # Create a video from the images
          create_video(data_root_path[i], savepath[i],model_name=model_names[m], top='top1')
          create_video(data_root_path[i], savepath[i],model_name=model_names[m], top='top5')

          print('total time per bg: ',time.time()-start)