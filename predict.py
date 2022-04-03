from numpy.lib.shape_base import _array_split_dispatcher
import torch
import pandas as pd
import numpy as np
from torchvision import models
import torch.nn as nn 

#-----------------加载模型---------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = './result/resnet18-30-64-adam-0.0001-pre-True.pth'
model = models.resnet18(pretrained=False).to(device)
cnn_features = model.fc.in_features
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
model.load_state_dict(torch.load(path))
model.to(device) 

#-----------------读入数据---------------------#

df_test_set = pd.read_json('./data/test/data/processed/test.json')
df_test_set['band_1'] = df_test_set['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
df_test_set['band_2'] = df_test_set['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
df_test_set['inc_angle'] = pd.to_numeric(df_test_set['inc_angle'], errors='coerce')
columns = ['id', 'is_iceberg']
df_pred = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)
# df_pred.id.astype(int)

for index, row in df_test_set.iterrows():
    rwo_no_id = row.drop('id')    
    b_1_test = (rwo_no_id['band_1']).reshape(-1, 75, 75)
    b_2_test = (rwo_no_id['band_2']).reshape(-1, 75, 75)
    b_3_test = (b_1_test + b_2_test)/2.0
    r = (b_1_test + abs(b_1_test.min())) / np.max((b_1_test + abs(b_1_test.min())))
    g = (b_2_test + abs(b_2_test.min())) / np.max((b_2_test + abs(b_2_test.min())))
    b = (b_3_test + abs(b_3_test.min())) / np.max((b_3_test + abs(b_3_test.min())))
    full_img_test = np.stack([r, g, b], axis=1)

    full_img_test = torch.from_numpy(full_img_test).type(torch.FloatTensor).to(device)
    # full_img_test = torch.squeeze(full_img_test, dim=0)   

    model.eval()
    # with torch.no_grad:       
    out = torch.squeeze(model(full_img_test.to(device))).cpu()
    p_pred = torch.softmax(out, dim=0)
    p = p_pred.detach().numpy()[1]
    
    df_pred = df_pred.append({'id':row['id'], 'is_iceberg':p},ignore_index=True)

df_pred.to_csv('./test.csv', index=False)