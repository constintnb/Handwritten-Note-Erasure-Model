import torch
from unet import UNet

def export(model_path, output_path):

    device = torch.device("cpu")
    model = UNet(3, 1).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    # 检查是否有 "_orig_mod." 前缀
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("_orig_mod."):
            name = k.replace("_orig_mod.", "") # 去掉前缀
        else:
            name = k
        new_state_dict[name] = v
    
    # 加载训练好的权重
    model.load_state_dict(new_state_dict)
    
    model.eval()

    # 创建一个假输入
    # ONNX 需要运行一次模型来追踪计算图，所以需要这个输入
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    # 执行导出
    torch.onnx.export(
        model,                  # 模型对象
        dummy_input,            # 假输入
        output_path,            # 输出文件名
        export_params=True,     # 导出权重
        opset_version=11,       # ONNX 版本 (11 兼容性最好)
        do_constant_folding=True, # 优化常量折叠
        input_names=['input'],  # 输入节点名称 (方便后续调用)
        output_names=['output'],# 输出节点名称
        # 动态轴：允许推理时输入不同尺寸的图片 (比如输入 1024x1024)
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"✅ 模型已成功导出为: {output_path}")

if __name__ == "__main__":
    export("model.pth", "model.onnx")