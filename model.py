import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd


    
def img_loader(self):
    img = Image.open(self.img_path)
    return img.convert("RGB")

def make_dataset(data_path,ans_path,alphabet):
    img_names = os.listdir(data_path)# 取出圖片名稱
    img_names.sort(key=lambda x: int(x.split(".")[0]))# 讓圖片從小到大排序
    df_ans = pd.read_csv(ans_path)# 讀取label的CSV檔
    ans_list = list(df_ans["code"].values)# 取得label
    samples = []

    # 用zip將圖片跟對應的答案湊一對
    for ans, img_name in zip(ans_list, img_names):
        if len(str(ans)) == 5  :#num_char:
            
            # 將圖片名稱及路徑合併
            # 以便上述程式img_loader的執行
            img_path = os.path.join(data_path, img_name)
            target = []
            # 這邊做5個字的辨識，例如：A5GG2
            # 會轉換成target = [0,31,6,6,28]
            for char in str(ans):
                pos = alphabet.find(char) 
                target.append(pos)
                
            # 用samples把他們全部包起來    
            samples.append((img_path, target))
        else:
            print(img_name)
        return samples  

class CaptchaData(Dataset):
    
    def __init__(self, data_path, ans_path, transform=None, target_transform=None):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.ans_path = ans_path
        # self.num_class = num_class
        # self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ012345678'
        self.samples = make_dataset(self.data_path, self.ans_path, self.alphabet)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)

def calculat_acc(output, target):
    output, target = output.view(-1, 36), target.view(-1, 36)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 5), target.view(-1, 5)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc