from unet import UNet
import torch
from torch.cuda.amp import autocast, GradScaler

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    run_loss = 0
    losses = []

    # FP16 加速
    scaler = GradScaler()   # 实例化Scalar

    for i, (imgs, masks) in enumerate(dataloader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        #开启混合精度上下文 (前向传播)
        # autocast(): 不影响模型准确率的前提下，自动把一部分运算的精度从“高精度（FP32）”降低到“半精度（FP16），提高速度
        with autocast():
            preds = model(imgs)
            loss = loss_fn(preds, masks)
        
        # 使用autocast之后，精度不够，产生梯度下溢问题
        # scaler：动态调整梯度大小
        scaler.scale(loss).backward()   #梯度放大65536倍
        scaler.step(optimizer)  # 还原梯度
        scaler.update() # 更新放大倍数（根据是否溢出）

        run_loss += loss.item()

        if (i+1)%10 == 0:
            avg_loss = run_loss / 10
            print(f"Step [{i+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")
            losses.append(avg_loss)
            run_loss = 0

    return losses

def test(dataloader, model, loss_fn, device):
    model.eval()

    size = len(dataloader.dataset)
    n = len(dataloader)
    loss = 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            with autocast():
                output = model(imgs)
                loss += loss_fn(output, masks).item()


    loss /= n

    print(f"测试结果: Avg loss: {loss:>8f} \n")
    return loss