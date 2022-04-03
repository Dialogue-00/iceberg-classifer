from enum import EnumMeta
import numpy as np
from tqdm import tqdm
import torch
import copy
import os

def loss_epoch(model, device, criterion, dataset_dl, opt=None):
    running_loss = 0.0
    running_corrects = 0.0
    len_data = len(dataset_dl.dataset)
    for i, data in enumerate(dataset_dl):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

           
        outputs = model(inputs)       
        predict = torch.softmax(outputs, dim=1)     # 输出值
        preds_class = outputs.argmax(dim=1)         # 取最大类别的概率
        loss = criterion(outputs, labels.long())

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
     
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds_class == labels.data).cpu()

    loss = running_loss / float(len_data)
    acc = running_corrects / float(len_data)

    return loss, acc





def train(device, model, params, save_path, train_dataloader, val_dataloader, criterion, optimizer, writer, scheduler=None):
    print('train start......')
    num_epochs = params.epochs
    step = 0
    # 存储每轮次损失值
    loss_history = {
        "train": [],
        "val": [],
    }   
    best_loss = float('inf')
    best_acc = 0.0
    # 存储每轮次正确数
    acc_history = {
        "train": [],
        "val": [],
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1,))
        print('-' * 10) 

        # 定义为模型训练阶段
        model = model.train()
        step += 1
        # with tqdm(total=train_size // params.batch_size) as pbar:
        train_loss, train_acc = loss_epoch(model,device,criterion,train_dataloader,optimizer)

        if scheduler is not None:
            scheduler.step()

        print("train loss: %.6f, acc: %.2f" %(train_loss,100*train_acc))
        # 存储训练各轮次结果值
        loss_history["train"].append(train_loss)
        acc_history["train"].append(train_acc)
        # 记录数据，保存于event file
        writer.add_scalars("Loss", {"Train": train_loss}, step)
        writer.add_scalars("Accuracy", {"Train": train_acc}, step)

        # 模型验证阶段    
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = loss_epoch(model,device,criterion,train_dataloader)

        # 存储过程中最好的结果数据
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            root = os.path.join(params.savedir + '/' + save_path)
            weight_path = os.path.join(root + '/' + save_path + '.pth')
            torch.save(model.state_dict(), weight_path)
            print("Copied best model weights!")

        print("val loss: %.6f, acc: %.2f" %(val_loss,100*val_acc))
        # 存储验证过程各轮次结果值
        loss_history["val"].append(val_loss)
        acc_history["val"].append(val_acc)

        writer.add_scalars("Loss", {"Valid": val_loss}, step)
        writer.add_scalars("Accuracy", {"Valid": val_acc}, step)
        

    # 加载整个过程中最好的参数
    model.load_state_dict(best_model_wts)
    train_loss = np.array(loss_history["train"])
    np.save(os.path.join(root + '/' + save_path), train_loss)

    return model, loss_history, acc_history

