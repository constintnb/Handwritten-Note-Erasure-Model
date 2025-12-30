from torch.utils.data import DataLoader
from data import Data
from unet import UNet
from train import train, test
from torch.utils.data import random_split
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    root = "D:\\deli\\20250211\\dataset"
    
    dataset = Data(root, (512,512))

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset= random_split(dataset, [train_size, test_size]) 

    train_loader = DataLoader(
        train_dataset,
        batch_size = 8, 
        shuffle = True, # 数据随机打乱
        num_workers = 4, 
        pin_memory = True,  # 开启内存到显存的高速传输通道
        persistent_workers = True,  # 不销毁进程（加速）
        drop_last = True    # 数据不足时自动丢弃
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = 8, 
        shuffle = False, 
        num_workers = 2, 
        pin_memory = True, 
        persistent_workers = False
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3,1).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float('inf')
    tot_loss = []
    epochs = 50

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
    