from diffusers import HunyuanDiT2DControlNetModel
from pipeline.hunyuan_controlnet import HunyuanDiTControlNetPipeline
import torch


controlnet = HunyuanDiT2DControlNetModel.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny", torch_dtype=torch.float16)

pipe = HunyuanDiTControlNetPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled", controlnet=controlnet, torch_dtype=torch.float16)
pipe.to("cuda")

from diffusers.utils import load_image
cond_image = load_image('https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny/resolve/main/canny.jpg?download=true')

## You may also use English prompt as HunyuanDiT supports both English and Chinese
prompt="在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围"
#prompt="At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming as if guarding the building. The background is the hotel entrance at night, with a close-up, eye-level, and centered composition. This photo presents a realistic photographic style, embodies Chinese sculpture culture, and reveals a mysterious atmosphere."

torch.manual_seed(42)
image = pipe(
    prompt,
    negative_prompt='错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，',
    height=1024,
    width=1024,
    guidance_scale=6.0,
    control_image=cond_image,
    num_inference_steps=50,
).images[0]

image.save('./image.png')
