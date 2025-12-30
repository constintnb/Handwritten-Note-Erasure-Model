import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class Data(Dataset):
    def __init__(self, root_dir, csize, repect=4):
        self.root_dir = root_dir     # 数据集的根目录
        self.in_dir = os.path.join(root_dir, "input")   # 原图
        self.out_dir = os.path.join(root_dir, "output")     # 效果图
        
        self.filename = sorted(os.listdir(self.in_dir))  # 建立一个有序的文件列表
        self.csize = csize # 裁剪大小
        self.repect = repect   # 同一张图被裁剪多少次

    def __len__(self):
        return len(self.filename) * self.repect

    def transform(self, image, mask):
        w, h = image.size   # 实际大小
        cw, ch = self.csize    # 裁剪大小

        dw = max(cw-w, 0)
        dh = max(ch-h, 0)

        # 图片大小不够，进行填充
        if dw > 0 or dh > 0:
            image = TF.pad(image, (0, 0, dw, dh), fill=0)
            mask = TF.pad(mask, (0, 0, dw, dh), fill=0)
            w,h = image.size

        # 随机裁剪
        x = random.randint(0, w-cw)
        y = random.randint(0, h-ch)
        image = TF.crop(image, y, x, ch, cw)
        mask = TF.crop(mask, y, x, ch, cw)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)


        return image, mask

    def __getitem__(self, idx):
        idx =  idx % len(self.filename)

        name = self.filename[idx]
        input_path = os.path.join(self.in_dir, name)
        output_path = os.path.join(self.out_dir, name)

        image = Image.open(input_path).convert("RGB")
        mask = Image.open(output_path).convert('L')

        image, mask = self.transform(image, mask)

        return image, mask

