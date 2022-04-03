import torch
from model.Alexnet import AlexNet
# from resnet_model import resnet34
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms


#ResNet
data_transformer = transforms.Compose([
transforms.ToPILImage(),
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.GaussianBlur(3),
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.ToTensor()
])

# model = resnet34(num_classes=2)
model = AlexNet(num_classes=2)

model_weight_path = './result/alexnet-epoch30-32-adam-0.0001-pre-True/alexnet-epoch30-32-adam-0.0001-pre-True.pth'
# model_weight_path = './result/alexnet-epoch30-32-adam-0.0001-pre-True/alexnet-epoch30-32-adam-0.0001-pre-True.pth'
model.load_state_dict(torch.load(model_weight_path))
# print(model)

img_path = './train/iceberg/iceberg_4.png'
img = Image.open(img_path).convert('RGB')
img = np.array(img, dtype=np.uint8)
img_tensor = data_transformer(img)
input_tensor = torch.unsqueeze(img_tensor, dim=0)

# forward
out_put = model(input_tensor)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(64):
        plt.subplot(8, 8, i+1)
        # [H, W, C]
        plt.imshow(im[:, :, i], cmap='gray')
        plt.axis('off')
    # plt.tight_layout()
    plt.show()
    plt.savefig('featuremap.png', dpi=300)
    plt.close()