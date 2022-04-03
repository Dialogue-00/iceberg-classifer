import pandas as pd
import numpy as np


data = pd.read_json('./data/train/data/processed/train.json')
b_1 = np.array(data.iloc[0]['band_1']).reshape(75, 75).astype(np.float32)
b_2 = np.array(data.iloc[0]['band_2']).reshape(75, 75).astype(np.float32)
b_3 = (b_1 + b_2) / 2.0
r = (b_1 + abs(b_1.min())) / np.max((b_1 + abs(b_1.min())))
g = (b_2 + abs(b_2.min())) / np.max((b_2 + abs(b_2.min()))) 
b = (b_3 + abs(b_3.min())) / np.max((b_3 + abs(b_3.min())))
full_img = np.stack([r, g, b], axis=2)
full_img*=255.0

x= 1
y=2