import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
import cv2
import numpy as np
import torch
import json

def horizontal_flip(image_array1, image_array2):
    if np.random.rand() < 0.5:  # 0.5
        flipped_image1 = np.flip(image_array1, axis=2)
        flipped_image2 = np.flip(image_array2, axis=2)
        return flipped_image1, flipped_image2
    else:
        return image_array1, image_array2

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, fold):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            fold: the five fold cross validate
        """
        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        data = self.generate_5Fold()
        self.A_paths = []
        self.B_paths = []
        self.labels = []
        self.label_range = {}  # store the range tha label related to
        boundry = 0
        
        for c in data:
            c_tot_len = 0
            if self.phase == 'train':  # get all part exclude the fold-number
                for i in range(5):
                    fold_len = c['low'][f'Fold{str(i+1)}']['len']
                    self.A_paths += c['low'][f'Fold{str(i+1)}']['data'] if i != fold else []
                    self.B_paths += c['high'][f'Fold{str(i+1)}']['data'] if i != fold else []
                    self.labels += [c['label']] * fold_len if i != fold else []
                    c_tot_len += fold_len if i != fold else 0
            else:
                fold_len = c['low'][f'Fold{str(fold+1)}']['len']
                self.A_paths += c['low'][f'Fold{str(fold+1)}']['data']
                self.B_paths += c['high'][f'Fold{str(fold+1)}']['data']
                self.labels += [c['label']] * fold_len
                c_tot_len = fold_len
            self.label_range[f"label{str(c['label'])}"] = [boundry, boundry + c_tot_len]
            boundry += c_tot_len

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # if opt.data_type == 'all':
        #     data_root = f'{opt.dataroot}'
        # else:
        #     data_root = f'{opt.dataroot}_{opt.data_type}'
        # self.dir_A = os.path.join(data_root, opt.phase + 'A')  # create a path '/dataroot/trainA'
        # self.dir_B = os.path.join(data_root, opt.phase + 'B')  # create a path '/dataroot/trainB'

        # self.A_paths = sorted(make_dataset(self.dir_A))   # load images from '/path/to/data/trainA'
        # self.B_paths = sorted(make_dataset(self.dir_B))    # load images from '/path/to/data/trainB'
        # self.label_path = os.path.join('../datasets', "label_inform.json") #TODO: For MTL Classification Labeling
        # self.A_size = len(self.A_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B
        # btoA = self.opt.direction == 'BtoA'
        # if opt.data_norm == 'ab_seperate':
        #     self.data_info_path = f"{data_root}/data.json"
        #     with open(self.data_info_path, "r") as json_file:
        #         loaded_data = json.load(json_file)
        #     print("Loaded Data:", loaded_data)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        label = self.labels[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs, but should be in same category
            low_boundry, high_boundry = self.label_range[f'label{str(label)}']
            index_B = random.randint(low_boundry, high_boundry - 1)
        B_path = self.B_paths[index_B]
        # print(A_path, B_path)
        # import time
        # time.sleep(2)
        
        A_img = cv2.imread(A_path, cv2.IMREAD_GRAYSCALE)
        if self.opt.resizeBig:
            A_img = cv2.resize(A_img, (512, 512))

        A_img_arr = np.array(A_img).reshape((1,) + A_img.shape)
        
        
        B_img = cv2.imread(B_path, cv2.IMREAD_GRAYSCALE)
        if self.opt.resizeBig:
            B_img = cv2.resize(B_img, (512, 512))
        B_img_arr = np.array(B_img).reshape((1,) + B_img.shape)

        if self.opt.data_norm == 'basic':
            A_img_arr_norm = ((A_img_arr / 255.) * 2) - 1
            B_img_arr_norm = ((B_img_arr / 255.) * 2) - 1
        else:
            raise NotImplementedError(self.opt.data_norm)
        # elif self.opt.data_norm == 'ab_seperate':
        #     with open(self.data_info_path, "r") as json_file:
        #         loaded_data = json.load(json_file)
        #     A_img_arr_norm = ((A_img_arr / 255.) - loaded_data['TrainA'][0]) / loaded_data['TrainA'][1]
        #     B_img_arr_norm = ((B_img_arr / 255.) - loaded_data['TrainB'][0]) / loaded_data['TrainB'][1]
        A = torch.from_numpy(A_img_arr_norm).float()
        B = torch.from_numpy(B_img_arr_norm).float()
        
        if self.phase == "train":
            if self.opt.lr_flip:
                A_img_arr, B_img_arr = horizontal_flip(A_img_arr, B_img_arr)
            
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}, label
        
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}, label

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
    
    def generate_5Fold(self):
        if not os.path.exists('./5fold.json'):
            data_json = []
            class_list = [i for i in os.listdir(self.opt.dataroot) if os.path.isdir(os.path.join(self.opt.dataroot, i))]  # ensure dir
            assert len(class_list) == 5
            for label, category in enumerate(class_list):
                A_path = os.path.join(self.opt.dataroot, category, 'low_quality')  # A
                B_path = os.path.join(self.opt.dataroot, category, 'high_quality')  # B
                A_path_list = [os.path.join(A_path, i) for i in sorted(os.listdir(A_path))]
                B_path_list = [os.path.join(B_path, i) for i in sorted(os.listdir(B_path))]
                import random
                compose = [(i, j) for i, j in zip(A_path_list, B_path_list)]
                random.shuffle(compose)
                A_path_list = [i[0] for i in compose]
                B_path_list = [i[1] for i in compose]
                assert len(A_path_list) == len(B_path_list)
                fold_length = len(A_path_list) // 5
                A_fold_list, B_fold_list = {}, {}
                for i in range(4):
                    A_fold_list[f'Fold{str(i+1)}'] = {'len': fold_length, 'data': A_path_list[fold_length * i: fold_length * (i+1)]}
                    B_fold_list[f'Fold{str(i+1)}'] = {'len': fold_length, 'data': B_path_list[fold_length * i: fold_length * (i+1)]}
                A_fold_list['Fold5'] = {'len': len(A_path_list) - fold_length * 4, 'data': A_path_list[fold_length * 4:]}
                B_fold_list['Fold5'] = {'len': len(A_path_list) - fold_length * 4, 'data': B_path_list[fold_length * 4:]}
                category_data = {
                    'label': label + 1,
                    'category': category,
                    'low': A_fold_list,
                    'high': B_fold_list
                }
                data_json.append(category_data)
            
            with open('./5fold.json', 'w', encoding='utf-8') as f:
                json.dump(data_json, f, ensure_ascii=False, indent=4)

        with open('./5fold.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data
    

