import glob
import os
import cv2
import config
import torch
import numpy as np
from torch import nn
from PIL import Image
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def predict_images(image_file, label, model):
    image = Image.open(image_file)
    image = image.convert("RGB")
    numpy_array = np.asarray(image.copy())
    image = transform_test(image)
    image = image.unsqueeze_(0).to(config.device)
    with torch.no_grad():
        outputs = model(image)
        outputs = outputs.to(config.device)
    predict_label = torch.max(outputs, dim=1)[1].data.cpu().numpy()[0]
    confidence = F.softmax(outputs[0], dim=0)
    probabilities = torch.max(confidence)
    probabilities = round(probabilities.item(), 3)
    print("image={},预测label={},概率={}".format(image_file, predict_label, probabilities))
    if predict_label != label:
        print("predict error image = {}".format(image_file))

    print("测试类别={}".format(predict_label))
    cv2.imshow("image", numpy_array)
    cv2.waitKey(0)


def get_image_label_to_predict():
    model = models.resnet18(pretrained=False)
    num_fits = model.fc.in_features
    model.fc = nn.Linear(num_fits, config.num_classes)
    model.load_state_dict(torch.load(config.predict_model)['net'])
    model.to(config.device)
    model.eval()
    classes_dir = os.listdir(config.predict_image_path)
    for label in classes_dir:
        label_path = os.path.join(config.predict_image_path, label)
        if os.path.isdir(label_path):
            images = glob.glob(os.path.join(label_path, "*.{}".format(config.image_format)))
            for img in images:
                predict_images(img, int(label), model)


if __name__ == '__main__':
    get_image_label_to_predict()
