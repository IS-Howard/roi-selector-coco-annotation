import os
import sys
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torch.nn.functional import upsample
import networks.deeplab_resnet as resnet #
from dataloaders import helpers as helpers #
from mypath import Path #
import glob
import configparser

def roi_click(image):
    modelName = 'dextr_pascal-sbd'
    pad = 50
    thres = 0.8
    gpu_id = 0
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
    clickTimes = 4

    #  Create the network and load the weights
    net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
    # print("Initializing weights from: {}".format(os.path.join(Path.models_dir(), modelName + '.pth')))
    state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), modelName + '.pth'),
                                    map_location=lambda storage, loc: storage)
    # Remove the prefix .module from the model when it is trained using DataParallel
    if 'module.' in list(state_dict_checkpoint.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict_checkpoint.items():
            name = k[7:]  # remove `module.` from multi-gpu training
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict_checkpoint
    net.load_state_dict(new_state_dict)
    net.eval()
    net.to(device)

    #  Read image and click the points
    plt.ion()
    plt.axis('off')
    plt.imshow(image)
    plt.title('Click the extreme points of the objects\nHit enter when done (do not close the window)')
    imgx = image.shape[0]
    imgy = image.shape[1]

    results = [] #list of the object region (bool)

    with torch.no_grad():
        while 1:
            extreme_points_ori = np.array(plt.ginput(clickTimes, timeout=0)).astype(np.int)
            if extreme_points_ori.shape[0] < clickTimes:
                break
        
                #sys.exit(0)
            boundp = np.array([min(extreme_points_ori[:,0])-20,min(extreme_points_ori[:,1])-20,
                            max(extreme_points_ori[:,0])+20,max(extreme_points_ori[:,1])+20])
            boundp[np.where(boundp < 0)] = 0
            if boundp[2] > imgx:
                boundp[2] = imgx
            if boundp[3] >imgy:
                boundp[3] = imgy
            #  Crop image to the bounding box from the extreme points and resize
            bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=pad, zero_pad=True) 
            crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
            resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

            #  Generate extreme point heat map normalized to image values
            extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad,
                                                                                                                        pad]
            extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
            extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
            extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

            #  Concatenate inputs and convert to tensor
            input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
            inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

            # Run a forward pass
            inputs = inputs.to(device)
            outputs = net.forward(inputs)
            outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
            outputs = outputs.to(torch.device('cpu'))

            pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres
            results.append(result)
            # Plot the results
            plt.imshow(helpers.overlay_masks(image / 255, results))
            plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')

    plt.close()
    plt.ioff()
    return np.array(results)

def save_roi_img(save_folder, image_folder, colors):
    colors = colors.split("/")[:-1]
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    #rename file of png to jpg
    for image_path in glob.glob(image_folder+"/*.png"):
        jpg = image_path.replace('.png','.jpg')
        os.rename(image_path, jpg)
    
    for image_path in glob.glob(image_folder+"/*.jpg"):
        print("processing image: ", image_path)
        image_path = image_path.replace('\\','/')
        image_name = image_path.split('/')[-1].replace('jpg', 'png')
        image = np.array(Image.open(image_path))
        results = roi_click(image)
        # plot roi with color
        for i in range(len(results)):
            result = results[i]
            if colors[i] == "0,0,0":
                continue
            color = tuple(map(int,colors[i].split(",")))
            mask1 = result
            new_img = np.zeros_like(image)
            new_img[np.where(mask1)] = color
            img_save = Image.fromarray(new_img.astype('uint8'))
            img_save.save(save_folder+'/'+image_name)

if __name__ == '__main__':
    parser = configparser.ConfigParser()
    parser.read("config.txt")
    image_folder = parser.get("config", "image_folder")
    save_folder = parser.get("config", "mask_folder")
    classes = parser.get("config", "classes")
    colors = parser.get("config", "colors")
    save_roi_img(save_folder,image_folder,colors)

