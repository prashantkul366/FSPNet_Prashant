# from torch.utils.data import Dataset
# import torch
# import numpy as np
# from imageio import imwrite, imread
# import os
# import torch.nn.functional as F
# import cv2
# # def img_normalize(image):
# #     if len(image.shape)==2:
# #         channel = (image[:, :, np.newaxis] - 0.485) / 0.229
# #         image = np.concatenate([channel,channel,channel], axis=2)
# #     else:
# #         image = (image-np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3)))\
# #                 /np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
# #     return image


# class TrainDataset(Dataset):
#     def __init__(self, paths):
#         self.image = []
#         self.label = []
#         self.count={}
#         for path in paths:
#             self.list = os.listdir(os.path.join(path, "Imgs"))
#             for i in self.list:
#                 self.image.append(os.path.join(path, "Imgs", i))
#                 self.label.append(os.path.join(path, "GT", i.split(".")[0] + ".png"))
#         print("Datasetsize:", len(self.image))

#     def __len__(self):
#         return len(self.image)
    
#     def __getitem__(self, item):
#         # img = imread(self.image[item]).astype(np.float32)/255.
#         img = np.load(self.image[item]).astype(np.float32)
#         # label = imread(self.label[item]).astype(np.float32)/255.
#         label = np.load(self.label[item]).astype(np.float32)

#         img = torch.from_numpy(img).permute(2,0,1)   # 4,H,W
#         label = torch.from_numpy(label).unsqueeze(0) # 1,H,W

#         ration = np.random.rand()
#         if ration<0.25:
#             img = cv2.flip(img, 1)
#             label = cv2.flip(label, 1)
#         elif ration<0.5:
#             img = cv2.flip(img, 0)
#             label = cv2.flip(label, 0)
#         elif ration<0.75:
#             img = cv2.flip(img, -1)
#             label = cv2.flip(label, -1)
#         if len(label.shape)==3:
#             label=label[:,:,0]
#         label=label[:,:,np.newaxis]

#         img = F.interpolate(img.unsqueeze(0), (384,384), mode='bilinear').squeeze(0)
#         label = F.interpolate(label.unsqueeze(0), (384,384), mode='nearest').squeeze(0)

#         return {"img": torch.from_numpy(img).permute(2,0,1).unsqueeze(0),
#                 "label":torch.from_numpy(label).permute(2,0,1).unsqueeze(0)}

# class TestDataset(Dataset):
#     def __init__(self, path, size):
#         self.size=size
#         self.image = []
#         self.label = []
#         self.list = os.listdir(os.path.join(path, "Imgs"))
#         self.count={}
#         for i in self.list:
#             self.image.append(os.path.join(path, "Imgs", i))
#             self.label.append(os.path.join(path, "GT", i.split(".")[0]+".png"))
#     def __len__(self):
#         return len(self.image)
#     def __getitem__(self, item):
#         img = imread(self.image[item]).astype(np.float32)/255.
#         label = imread(self.label[item]).astype(np.float32)/255.
#         if len(label.shape)==2:
#             label=label[:,:,np.newaxis]
#         return {"img": F.interpolate(torch.from_numpy(img_normalize(img)).permute(2,0,1).unsqueeze(0), (self.size, self.size), mode='bilinear', align_corners=True).squeeze(0),
#                 "label": torch.from_numpy(label).permute(2,0,1),
#                 'name': self.label[item]}

# def my_collate_fn(batch):
#     size = 384
#     imgs=[]
#     labels=[]
#     for item in batch:
#         imgs.append(F.interpolate(item['img'], (size, size), mode='bilinear'))
#         labels.append(F.interpolate(item['label'], (size, size), mode='bilinear'))
#     return {'img': torch.cat(imgs, 0),
#             'label': torch.cat(labels, 0)}


from torch.utils.data import Dataset
import torch
import numpy as np
import os
import torch.nn.functional as F


class TrainDataset(Dataset):

    def __init__(self, root):

        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")

        self.files = sorted(os.listdir(self.img_dir))

        print("===================================")
        print("TRAIN DATASET INITIALIZED")
        print("Image path :", self.img_dir)
        print("Mask path  :", self.mask_dir)
        print("Dataset size :", len(self.files))
        print("===================================")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # img = np.load(os.path.join(self.img_dir, self.files[idx])).astype(np.float32)
        # mask = np.load(os.path.join(self.mask_dir, self.files[idx])).astype(np.float32)

        # # -------- CHANNEL WISE NORMALIZATION --------
        # for c in range(img.shape[2]):

        #     mean = img[:,:,c].mean()
        #     std  = img[:,:,c].std()

        #     img[:,:,c] = (img[:,:,c] - mean) / (std + 1e-6)
        # # ---------- VERY IMPORTANT ----------
        # # fix channel order if stored wrongly
        # # some numpy saved as C,H,W
        # if img.shape[0] == 4:
        #     img = np.transpose(img, (1,2,0))   # -> H W 4

        # # # ---------- normalize hillshade ----------
        # # img = (img - img.mean()) / (img.std() + 1e-6)

        # # ---------- make binary mask ----------
        # mask = (mask > 0).astype(np.float32)

        # # ---------- tensor ----------
        # img = torch.from_numpy(img).permute(2,0,1)   # 4 H W
        # mask = torch.from_numpy(mask).unsqueeze(0)   # 1 H W

        # # ---------- resize ----------
        # img = F.interpolate(img.unsqueeze(0), (384,384), mode='bilinear', align_corners=False).squeeze(0)
        # mask = F.interpolate(mask.unsqueeze(0), (384,384), mode='nearest').squeeze(0)

        img = np.load(os.path.join(self.img_dir, self.files[idx])).astype(np.float32)
        mask = np.load(os.path.join(self.mask_dir, self.files[idx])).astype(np.float32)

        # ⭐ FIRST fix layout
        if img.shape[0] == 4:
            img = np.transpose(img, (1,2,0))   # H W C

        # ⭐ THEN channel-wise normalization
        for c in range(img.shape[2]):

            mean = img[:,:,c].mean()
            std  = img[:,:,c].std()

            img[:,:,c] = (img[:,:,c] - mean) / (std + 1e-6)

        # binary mask
        mask = (mask > 0).astype(np.float32)

        # tensor
        img = torch.from_numpy(img).permute(2,0,1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        # resize
        img = F.interpolate(img.unsqueeze(0),(384,384),mode="bilinear",align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0),(384,384),mode="nearest").squeeze(0)
        return {"img": img, "label": mask}
    
    
def my_collate_fn(batch):

    imgs = torch.stack([item["img"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    return {
        "img": imgs,
        "label": labels
    }