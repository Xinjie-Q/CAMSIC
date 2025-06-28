import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
from typing import Dict
from torch import Tensor
import numpy as np
import glob
import json
from PIL import Image
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F

def print_text(text, local_rank):
    if local_rank == 0:
        print(text)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class CropCityscapesArtefacts:
    """Crop Cityscapes images to remove artefacts"""

    def __init__(self):
        self.top = 64
        self.left = 128
        self.right = 128
        self.bottom = 256

    def __call__(self, image):
        """Crops a PIL image.
        Args:
            image (PIL.Image): Cityscapes image (or disparity map)
        Returns:
            PIL.Image: Cropped PIL Image
        """
        w, h = image.size
        assert w == 2048 and h == 1024, f'Expected (2048, 1024) image but got ({w}, {h}). Maybe the ordering of transforms is wrong?'
        #w, h = 1792, 704
        return image.crop((self.left, self.top, w-self.right, h-self.bottom))

class MinimalCrop:
    """
    Performs the minimal crop such that height and width are both divisible by min_div.
    """
    
    def __init__(self, min_div=16):
        self.min_div = min_div
        
    def __call__(self, image):
        w, h = image.size
        
        h_new = h - (h % self.min_div)
        w_new = w - (w % self.min_div)
        
        if h_new == 0 and w_new == 0:
            return image
        else:    
            h_diff = h-h_new
            w_diff = w-w_new

            top = int(h_diff/2)
            bottom = h_diff-top
            left = int(w_diff/2)
            right = w_diff-left

            return image.crop((left, top, w-right, h-bottom))

class StereoImageDataset(Dataset):
    """Dataset class for image compression datasets."""
    #/home/xzhangga/datasets/Instereo2K/train/
    def __init__(self, ds_type='train', ds_name='cityscapes', root='/home/xzhangga/datasets/Cityscapes/', crop_size=(256, 256), **kwargs):
        """
        Args:
            name (str): name of dataset, template: ds_name#ds_type. No '#' in ds_name or ds_type allowed. ds_type in (train, eval, test).
            path (str): if given the dataset is loaded from path instead of by name.
            transforms (Transform): transforms to apply to image
            debug (bool, optional): If set to true, limits the list of files to 10. Defaults to False.
        """
        super().__init__()
        
        self.path = Path(f"{root}")
        self.ds_name = ds_name
        self.ds_type = ds_type
        if ds_type=="train":
            # self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size), 
            #     transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),]) 
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(crop_size)]) 
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.left_image_list, self.right_image_list = self.get_files()


        # if ds_name == 'cityscapes':
        #     self.crop = CropCityscapesArtefacts()
        # else:
            # if ds_type == "test" and ds_name!="instereo512":
            #     self.crop = MinimalCrop(min_div=64)
            # else:
        self.crop = None
        #self.index_count = 0

        print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.left_image_list)} files.')

    def __len__(self):
        return len(self.left_image_list)

    def __getitem__(self, index):
        #self.index_count += 1
        image_list = [Image.open(self.left_image_list[index]).convert('RGB'), Image.open(self.right_image_list[index]).convert('RGB')]
        if self.crop is not None:
            image_list = [self.crop(image) for image in image_list]
        frames = np.concatenate([np.asarray(image) for image in image_list], axis=-1)
        frames = torch.chunk(self.transform(frames), 2)
        # if random.random() < 0.5:
        #     frames = frames[::-1]
        return frames

    def get_files(self):
        if self.ds_name == 'cityscapes':
            left_image_list, right_image_list, disparity_list = [], [], []
            for left_image_path in self.path.glob(f'leftImg8bit/{self.ds_type}/*/*.png'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("leftImg8bit", 'rightImg8bit'))
                # disparity_list.append(str(left_image_path).replace("leftImg8bit", 'disparity'))
            if self.ds_type=="test":
                left_image_list, right_image_list = left_image_list[:20], right_image_list[:20]

        elif self.ds_name == 'instereo2k':
            path = self.path / self.ds_type
            if self.ds_type == "test":
                folders = [f for f in path.iterdir() if f.is_dir()]
            else:
                folders = [f for f in path.glob('*/*') if f.is_dir()]
            left_image_list = [f / 'left.png' for f in folders]
            right_image_list = [f / 'right.png' for f in folders]
        
        elif self.ds_name == 'instereo512':
            path = self.path / self.ds_type
            left_image_list, right_image_list = [], []
            for left_image_path in path.glob(f'left/*'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("left", 'right'))

        elif self.ds_name == 'holopix50k':
            if self.ds_type == "test":
                path = Path("/home/xzhangga/dataset/instereo2k/") / self.ds_type
                folders = [f for f in path.iterdir() if f.is_dir()]
                left_image_list = [f / 'left.png' for f in folders]
                right_image_list = [f / 'right.png' for f in folders]
            else:
                left_image_list, right_image_list = [], []
                for left_image_path in self.path.glob(f'*/left/*'):
                    left_image_list.append(str(left_image_path))
                    right_image_list.append(str(left_image_path).replace("left", 'right'))
                #print(len(left_image_list))

        elif self.ds_name == 'kitti':
            left_image_list, right_image_list = [], []
            path = self.path / self.ds_type
            left_image_list, right_image_list = [], []
            for left_image_path in path.glob(f'left/*'):
                left_image_list.append(str(left_image_path))
                right_image_list.append(str(left_image_path).replace("left", 'right'))

        elif self.ds_name == 'wildtrack':
            C1_image_list, C4_image_list = [], []
            for image_path in self.path.glob(f'images/C1/*.png'):
                if self.ds_type == "train" and int(image_path.stem) <= 2000:
                    C1_image_list.append(str(image_path))
                    C4_image_list.append(str(image_path).replace("C1", 'C4'))
                elif self.ds_type == "test" and int(image_path.stem) > 2000:
                    C1_image_list.append(str(image_path))
                    C4_image_list.append(str(image_path).replace("C1", 'C4'))
            left_image_list, right_image_list = C1_image_list, C4_image_list
        else:
            raise NotImplementedError

        return left_image_list, right_image_list


def pad(x, p: int = 2 ** (4 + 2)):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    x = F.pad(
        x,
        padding,
        mode="constant",
        value=0,
    )
    return x, padding

def crop(x, padding):
    return F.pad(x, tuple(-p for p in padding))

class SingleStereoImageDataset(StereoImageDataset):
    """Dataset class for image compression datasets."""
    #/home/xzhangga/datasets/Instereo2K/train/
    def __init__(self, ds_type='train', ds_name='cityscapes', root='/home/xzhangga/datasets/Cityscapes/', crop_size=(256, 256), **kwargs):
        """
        Args:
            name (str): name of dataset, template: ds_name#ds_type. No '#' in ds_name or ds_type allowed. ds_type in (train, eval, test).
            path (str): if given the dataset is loaded from path instead of by name.
            transforms (Transform): transforms to apply to image
            debug (bool, optional): If set to true, limits the list of files to 10. Defaults to False.
        """
        super().__init__(ds_type, ds_name, root, crop_size)
        self.image_list = self.left_image_list + self.right_image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        #self.index_count += 1
        frames = Image.open(self.image_list[index]).convert('RGB')
        frames = self.transform(frames)
        return frames

class FLIRDataset(Dataset):
    def __init__(self, ds_type='train', ds_name='cityscapes', root='/home/xzhangga/datasets/Cityscapes/', crop_size=(256, 256), **kwargs):
        super().__init__()
        self.path = Path(f"{root}")
        self.ds_name = ds_name
        self.ds_type = ds_type
        if ds_type=="train":
            self.transform = transforms.Compose([transforms.ToTensor()])
        else: 
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.rgb_image_list, self.infrared_image_list = self.get_files()

        print(f'Loaded dataset {ds_name} from {self.path}. Found {len(self.rgb_image_list)} files.')

    def __len__(self):
        return len(self.rgb_image_list)

    def __getitem__(self, index):
        #self.index_count += 1
        rgb_img = Image.open(self.rgb_image_list[index]).convert('RGB')
        infrared_img = Image.open(self.infrared_image_list[index]).convert('L')
        infrared_img, rgb_img = self.transform(infrared_img), self.transform(rgb_img)
        return [infrared_img, rgb_img]

    def get_files(self):
        if self.ds_name == 'flir':
            ir_image_list, rgb_image_list = [], []
            for rgb_image_path in self.path.glob(f'{self.ds_type}/RGB/*'):
                rgb_image_list.append(str(rgb_image_path))
                ir_image_list.append(str(rgb_image_path).replace("RGB", 'IR'))
        else:
            raise NotImplementedError

        return rgb_image_list, ir_image_list

class InfraredImageDataset(Dataset):
    def __init__(self, root, ds_type='train', transform=None):
        self.path = Path(f"{root}")
        self.ds_type=ds_type
        self.samples = [str(f) for f in self.path.glob(f'{self.ds_type}/IR/*')]
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.samples[index]
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.samples)



def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


def get_files_list(dir, name):
    '''
    实现遍历dir目录下,所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名   

    classes = json.load(open(os.path.join(dir, "Labels.json")))
    classes_dict = {}
    for i, key in enumerate(classes.keys()):
        classes_dict[key] = i
    files_list = [] #写入文件的数据
    if name == 'train':    
        for i in range(4):
            filepath = os.path.join(dir, name + '.X{}'.format(i+1))
            for parent, dirnames, filenames in os.walk(filepath):
                for filename in filenames:
                    curr_file = parent.split(os.sep)[-1]    #获取正在遍历的文件夹名（也就是类名）
                    labels = classes_dict[curr_file]
                    files_list.append([os.path.join(parent, filename).replace('\\','/'), labels])    #相对路径+label
    else:
        filepath = os.path.join(dir, name + '.X')
        for parent, dirnames, filenames in os.walk(filepath):
            for filename in filenames:
                curr_file = parent.split(os.sep)[-1]    #获取正在遍历的文件夹名（也就是类名）
                labels = classes_dict[curr_file]
                files_list.append([os.path.join(parent, filename).replace('\\','/'), labels])    #相对路径+label
    return files_list




def save_checkpoint(state, is_best=False, log_dir=None, filename="ckpt.pth.tar"):
    save_file = os.path.join(log_dir, filename)
    print("save model in:", save_file)
    torch.save(state, save_file)
    if is_best:
        torch.save(state, os.path.join(log_dir, filename.replace(".pth.tar", ".best.pth.tar")))


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def rename_key(key: str, num_model: int) -> str:
    """Rename state_dict key."""

    # Deal with modules trained with DataParallel
    #startswith() 方法用于检查字符串是否是以指定子字符串开头
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    for stage in range(num_model):
        if key.startswith(f"entropy_bottleneck_{stage:d}."):
            if key.startswith(f"entropy_bottleneck_{stage:d}._biases."):
                return f"entropy_bottleneck_{stage:d}._bias{key[-1]}"

            if key.startswith(f"entropy_bottleneck_{stage:d}._matrices."):
                return f"entropy_bottleneck_{stage:d}._matrix{key[-1]}"

            if key.startswith(f"entropy_bottleneck_{stage:d}._factors."):
                return f"entropy_bottleneck_{stage:d}._factor{key[-1]}"

    return key


def load_state_dict(state_dict: Dict[str, Tensor], num_model: int) -> Dict[str, Tensor]:
    """Convert state_dict keys."""
    state_dict = {rename_key(k, num_model): v for k, v in state_dict.items()}
    return state_dict



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class MultipleAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, width_list, fmt=':f'):
        for i, width in enumerate(width_list):
            setattr(self, name+"_"+str(width), AverageMeter(name+"_"+str(width), fmt))
        self.name = name
        self.fmt = fmt

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def get_output_folder(parent_dir, env_name, output_current_folder=False):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    if not output_current_folder: 
        experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir, experiment_id



if __name__ == '__main__':

    #import pandas as pd
    #先生成两个csv文件夹
    #df = pd.DataFrame(columns=['path', 'label'])
    #df.to_csv("./Dataset/train.csv", index=False)

    #df2 = pd.DataFrame(columns=['path', 'label'])
    #df2.to_csv("./Dataset/val.csv", index=False)

    #写入txt文件
    train_dir = '/home/xzhangga/datasets/imagenet100'
    train_txt = '/home/xzhangga/datasets/imagenet100/train.txt'
    train_data = get_files_list(train_dir, name='train')
    write_txt(train_data, train_txt, mode='w')

    val_dir = '/home/xzhangga/datasets/imagenet100'
    val_txt = '/home/xzhangga/datasets/imagenet100/val.txt'
    val_data = get_files_list(val_dir, name='val')
    write_txt(val_data, val_txt, mode='w')


'''

def get_files_list(dir, name):

    实现遍历dir目录下,所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list

    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名   

    classes = json.load(open(os.path.join(dir, "Labels.json")))
    classes_dict = {}
    for i, key in enumerate(classes.keys()):
        classes_dict[key] = i+1
    files_list = [] #写入文件的数据
    if name == 'train':    
    for i in range(4):
        filepath = os.path.join(dir, name + '.X{}'.format(i+1))
        for parent, dirnames, filenames in os.walk(filepath):
            for filename in filenames:
                #print("parent is: " + parent)
                #print("filename is: " + filename)
                #print(os.path.join(parent, filename).replace('\\','/'))  # 输出rootdir路径下所有文件（包含子文件）信息
                curr_file = parent.split(os.sep)[-1]    #获取正在遍历的文件夹名（也就是类名）
                #print("curr_file is: " + curr_file)
                labels = classes_dict[curr_file]
                #print("labels:", labels)
                #dir_path = parent.replace('\\', '/').split('/')[-2]   #train?val?test?
                #curr_file = os.path.join(dir_path, curr_file)  #相对路径
                #print(curr_file)
                #print(os.path.join(parent, filename).replace('\\','/'))
                #input()

                files_list.append([os.path.join(parent, filename).replace('\\','/'), labels])    #相对路径+label

             
                #写入csv文件
                path = "%s" % os.path.join(curr_file, filename).replace('\\','/')
                label = "%d" % labels
                list = [path, label]
                data = pd.DataFrame([list])
                if dir == './Dataset/train':
                    data.to_csv("./Dataset/train.csv", mode='a', header=False, index=False)
                elif dir == './Dataset/val':
                    data.to_csv("./Dataset/val.csv", mode='a', header=False, index=False)
        
    return files_list
'''