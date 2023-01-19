import cv2
import numpy as np
import torch
from . import RRDBNet_arch as arch

async def generate_SR_image(src_image, gpu_override=False):
    if not gpu_override:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model_path = "esrgan_app/components/esrgan/models/RRDB_ESRGAN_x4.pth"
    
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    img = cv2.imdecode(np.fromstring(await src_image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    
    torch.cuda.empty_cache()
    del device, model_path, model, img

    return output


