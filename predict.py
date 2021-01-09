import torch
from model import vgg
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os

path1 = "C://Python_Files/vggnet/flower_data/val/daisy"   #图像读取地址
path2 = "C://Python_Files/vggnet/flower_data/val/dandelion"
path3 = "C://Python_Files/vggnet/flower_data/val/roses"
path4 = "C://Python_Files/vggnet/flower_data/val/sunflowers"
path5 = "C://Python_Files/vggnet/flower_data/val/tulips"
filelist1 = os.listdir(path1)  # 打开对应的文件夹
filelist2 = os.listdir(path2)
filelist3 = os.listdir(path3)
filelist4 = os.listdir(path4)
filelist5 = os.listdir(path5)
total_num1 = len(filelist1)  #得到文件夹中图像的个数
total_num2 = len(filelist2)
total_num3 = len(filelist3)
total_num4 = len(filelist4)
total_num5 = len(filelist5)
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
count1 = count2 = count3 = count4 = count5 = 0
# 预测开始
for i in range(89):
    if i < total_num1:
        img = Image.open("C://Python_Files/vggnet/flower_data/val/daisy/" + str(i + 1) + ".jpg")
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        try:
            json_file = open('./class_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)
        model = vgg(model_name="vgg16", num_classes=5)
        model_weight_path = "./vgg16Net_40.pth"
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        if class_indict[str(predict_cla)] == "daisy":
            count1 = count1 + 1
    if i < total_num2:
        img = Image.open("C://Python_Files/vggnet/flower_data/val/dandelion/" + str(i + 1) + ".jpg")
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        try:
            json_file = open('./class_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)
    # create model
        model = vgg(model_name="vgg16", num_classes=5)
    # load model weights
        model_weight_path = "./vgg16Net_40.pth"#训练好的模型文件
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        if class_indict[str(predict_cla)] == "dandelion":
            count2 = count2 + 1
    if i < total_num3:
        img = Image.open("C://Python_Files/vggnet/flower_data/val/roses/" + str(i + 1) + ".jpg")#文件夹里的文件
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        try:
            json_file = open('./class_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)
        model = vgg(model_name="vgg16", num_classes=5)
        model_weight_path = "./vgg16Net_40.pth"
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        if class_indict[str(predict_cla)] == "roses":
                count3 = count3 + 1
    if i < total_num4:
        img = Image.open("C://Python_Files/vggnet/flower_data/val/sunflowers/" + str(i + 1) + ".jpg")
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        try:
            json_file = open('./class_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)
        model = vgg(model_name="vgg16", num_classes=5)
        model_weight_path = "./vgg16Net_40.pth"
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        if class_indict[str(predict_cla)] == "sunflowers":
            count4 = count4 + 1
    if i < total_num5:
        img = Image.open("C://Python_Files/vggnet/flower_data/val/tulips/" + str(i + 1) + ".jpg")
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        try:
            json_file = open('./class_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)
        model = vgg(model_name="vgg16", num_classes=5)
        model_weight_path = "./vgg16Net_40.pth"
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        if class_indict[str(predict_cla)] == "tulips":
            count5 = count5 + 1
avg1 = count1 / total_num1
avg2 = count2 / total_num2
avg3 = count3 / total_num3
avg4 = count4 / total_num4
avg5 = count5 / total_num5

data = open("./data_40.txt", 'w+')
print(avg1, avg2, avg3, avg4, avg5, file=data)
data.close()

y = [0, avg1, avg2, avg3, avg4, avg5]#这里与下面的0均是原点，作为比较
x = ["0", "daisy", "dandelion", "roses", "sunflowers", "tulips"]
plt.scatter(x, y, s=100)

plt.title("The accuracy of 5 flowers", fontsize=22)
plt.xlabel("Flowers", fontsize=16)
plt.ylabel("Accuracy", fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=10)
plt.savefig('./pic/pre_40.jpg')
plt.show()