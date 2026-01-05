from data import Data
from unet import UNet
from train import train, test
from torch.utils.data import random_split, DataLoader, ConcatDataset
import torch
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    roots = [
        "/root/autodl-tmp/deli/20250211/dataset",
        "/root/autodl-tmp/deli/20250212/dataset",
        "/root/autodl-tmp/deli/20250213/dataset"
    ]
    
    datasets = []

    for r in roots:
        if os.path.exists(r):
            ds = Data(r, (512, 512), cache_mode=True)
            datasets.append(ds)

    full_dataset = ConcatDataset(datasets)

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset= random_split(full_dataset, [train_size, test_size]) 

    train_loader = DataLoader(
        train_dataset,
        batch_size = 24, 
        shuffle = True, # 数据随机打乱
        num_workers = 8, 
        #prefetch_factor = 8,  # 预取
        pin_memory = True,  # 开启内存到显存的高速传输通道
        persistent_workers = True,  # 不销毁进程（加速）
        drop_last = True    # 数据不足时自动丢弃
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = 24, 
        shuffle = False, 
        num_workers = 8, 
        pin_memory = True, 
        persistent_workers = False
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3,1).to(device)

    model = torch.compile(model) # 加速模型

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float('inf')
    tot_loss = []
    epochs = 35

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # 调用训练函数 
        train_losses = train(train_loader, model, loss_fn, optimizer, device)
        tot_loss.extend(train_losses)
        
        # 调用测试函数 
        loss = test(test_loader, model, loss_fn, device)

        # 保存损失最低的模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "model.pth")
            print("模型已保存。")

    # 绘制Loss图像
    plt.figure(figsize=(10, 5))
    plt.plot(tot_loss, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Steps (per 100 batches)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    