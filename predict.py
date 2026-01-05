import torch
from unet import UNet
from PIL import Image
import torchvision.transforms.functional as TF

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3,1).to(device)

    model_path = "model.pth"
    state_dict = torch.load(model_path, map_location=device)


    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            name = k.replace("_orig_mod.", "")  # 去掉前缀
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()

    img_path = "D:\\deli\\20250212\\dataset\\input\\7288931689241272321.jpg"
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    
    img_tensor = TF.to_tensor(img)
    #img_tensor = TF.normalize(img_tensor, [0.5,0.5,0.5], [0.5,0.5,0.5])
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    output = output.squeeze(0)
    out_img = TF.to_pil_image(output.cpu())
    out_img.save("result.jpg")
    print(f"Saved result to result.jpg")