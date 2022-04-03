from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GuidedBackpropReLUModel
from torchvision.models import resnet50
import numpy as np
from torchvision.transforms.functional import normalize
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn 
import cv2
def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-----------------加载模型---------------------#
path = './result/resnet50-30-32-adam-0.0001-pre-False.pth'
model = models.resnet50(pretrained=False).to(device)
cnn_features = model.fc.in_features
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
model.load_state_dict(torch.load(path))
model.to(device) 
target_layers = [model.layer4[-1]]

#-----------------读入数据---------------------#
data_transform = transforms.Compose([transforms.ToTensor()])
# img_path = './train/ship/ship_844.png'
img_path = './train/iceberg/iceberg_4.png'
img = Image.open(img_path).convert('RGB')
img = np.array(img, dtype=np.uint8)
img_tensor = data_transform(img)
input_tensor = torch.unsqueeze(img_tensor, dim=0)

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
target_category = None

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category,aug_smooth=True)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)

plt.imshow(cam_image)
plt.show()
plt.axis('off') 
plt.savefig('iceberg_4_grad++.png', dpi=300)

# cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


# gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
# gb = gb_model(input_tensor)

# cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
# cam_gb = deprocess_image(cam_mask * gb)
# gb = deprocess_image(gb)

# cv2.imwrite(f'cam.jpg', cam_image)
# cv2.imwrite(f'gb.jpg', gb)
# cv2.imwrite(f'cam_gb.jpg', cam_gb)

