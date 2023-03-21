import sys
import os
import cv2
import importlib
import numpy as np
import argparse
import yaml
import torchvision.transforms as transforms
import torch
sys.path.append(os.path.dirname(sys.path[0]))


def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(imgs):
    """
    Input: t,c,h,w
    """
    def _toimg(img):

        img = torch.clamp(img, 0, 1)
        img = img.numpy().transpose(1, 2, 0)

        img = (img * 255.0).round().astype('uint8')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if isinstance(imgs, list):
        return [_toimg(img) for img in imgs]
    else:
        return _toimg(imgs)    


def Load_model(opts, config_dict):
    net = importlib.import_module('VP_code.models.' + opts.model_name)
    netG = net.Video_Backbone()

    if opts.pretrained:
        model_path = opts.pretrained
    else:
        model_path = os.path.join('OUTPUT', opts.name, 'models', 'net_G_{}.pth'.format(str(opts.which_iter).zfill(5)))
    
    checkpoint = torch.load(model_path)
    netG.load_state_dict(checkpoint['netG'])
    netG.cuda()
    print("Finish loading model ...")
    return netG


def inference(opts, input_video_url, config_dict, loaded_model):
    print("Input Video URL: ", input_video_url)
    video_name_list = input_video_url.strip("").split("/")
    video_name = video_name_list[-1] if video_name_list[-1] != '' else video_name_list[-2]
    
    save_path = os.path.join(opts.save_place, video_name)
    print("Video name: ", video_name, "Save path: ", save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    loaded_model.eval()

    print("Loading Frame Count...")
    frame_list = os.listdir(input_video_url)
    frame_list = [w for w in frame_list if w.endswith(".png") or w.endswith(".jpg")]
    all_len = len(frame_list)
    print("Frame count: ", all_len)

    val_frame_num = config_dict['val']['val_frame_num']
    for i in range(1, all_len, opts.temporal_stride):
        # if i > 10: break

        print("Current: {} | Frame Count: {}".format(i, all_len))
        img_lqs = []

        for j in range(i, min(i + val_frame_num, all_len) + 1):
            img_path = os.path.join(input_video_url, "f{:03d}.png".format(j))
            img_lq = cv2.imread(img_path)
            img_lq = img_lq.astype(np.float32) / 255.
            img_lqs.append(img_lq)
        
        img_results = img2tensor(img_lqs) ## List of tensor

        transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        for k in range(len(img_results)):
            img_results[k] = transform_normalize(img_results[k])
        
        img_lqs = torch.stack(img_results, dim=0).unsqueeze(0).cuda()
        
        with torch.no_grad():
            output_results = loaded_model(img_lqs)
            output_results = (output_results + 1.) / 2.
        
        output_results = output_results.squeeze(0).cpu()
        t, c, h, w = output_results.size()

        for j in range(t):
            img = tensor2img(output_results[j])
            frame_save_place = os.path.join(save_path, "f{:03d}.png".format(i + j))
            cv2.imwrite(frame_save_place, img)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='BRT_tlc',help='The name of this experiment')
    parser.add_argument('--model_name',type=str,default='BRT_tlc',help='The name of adopted model')
    parser.add_argument('--input_video_url',type=str,default='/data1/ljj/NTIRE2023/Colorization/DATA/test_frames/test',help='degraded video input')
    parser.add_argument('--temporal_length',type=int,default=8,help='How many frames should be processed in one forward')
    parser.add_argument('--temporal_stride',type=int,default=7,help='Stride value while sliding window')
    parser.add_argument('--save_place',type=str,default='output_dir/fid_results',help='save place')
    opts = parser.parse_args()

    with open(os.path.join('./configs', opts.name+'.yaml'), 'r') as stream:
        config_dict = yaml.safe_load(stream)

    config_dict['datasets']['val']['dataroot_lq'] = opts.input_video_url
    config_dict['val']['val_frame_num'] = opts.temporal_length

    clip_list = os.listdir(opts.input_video_url)
    clip_list.sort()

    if not os.path.exists(opts.save_place):
        os.makedirs(opts.save_place, exist_ok=True)

    # test phase class
    part1_list = ["001", "009", "012", "015"]
    part2_list = ["002", "004", "006", "007", "010"]
    part3_list = ["005", "008", "014"]
    part4_list = ["011"]
    part5_list = ["013"]
    other_list = ["003"]

    for index, clip in enumerate(clip_list):
        print("Part 1: {} | {}".format(index + 1, len(clip_list)))
        input_video_url = os.path.join(opts.input_video_url, clip)

        if clip in part1_list:
            opts.pretrained = "weight/part1/net_G_07000.pth"
        elif clip in part2_list:
            opts.pretrained = "weight/part2/net_G_09000.pth"
        elif clip in part3_list:
            opts.pretrained = "weight/part3/net_G_05000.pth"
        elif clip in part4_list:
            opts.pretrained = "weight/part4/net_G_01500.pth"    
        elif clip in part5_list:
            opts.pretrained = "weight/part5/net_G_36000.pth"
        elif clip in other_list:
            opts.pretrained = "weight/net_G_04000_base.pth"
        
        base_model = Load_model(opts, config_dict)
        inference(opts, input_video_url, config_dict, base_model)

    print("temporal_length: {} | temporal_stride: {}".format(opts.temporal_length, opts.temporal_stride))
    print("{} finished.".format(opts.save_place))