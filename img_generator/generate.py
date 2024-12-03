# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import json
import sys 
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
print(root_dir)

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import argparse
import shutil
from collections import defaultdict
import torchvision
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer

from diffusers.utils.import_utils import is_xformers_available

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from cp_dataset import VitonHDTestDataset



logger = get_logger(__name__, log_level="INFO")



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default= "yisol/IDM-VTON",required=False,)
    parser.add_argument("--width",type=int,default=768,)
    parser.add_argument("--height",type=int,default=1024,)
    parser.add_argument("--num_inference_steps",type=int,default=30,)
    parser.add_argument("--output_dir",type=str,default="result",)
    parser.add_argument("--unpaired",action="store_true",)
    parser.add_argument("--data_dir",type=str,default="/home/omnious/workspace/yisol/Dataset/zalando")
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--test_batch_size", type=int, default=2,)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--data_list", type=str, default="subtest_0.1.txt")
    parser.add_argument("--phase", type=str, default="test")
    args = parser.parse_args()


    return args

def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images

def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            if os.path.exists(args.output_dir):
                shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)

    weight_dtype = torch.float16
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    #     args.mixed_precision = accelerator.mixed_precision
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    #     args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )


    # Freeze vae and text_encoder and set unet to trainable
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    UNet_Encoder.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    UNet_Encoder.to(accelerator.device, weight_dtype)
    unet.eval()
    UNet_Encoder.eval()

    
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    test_dataset = VitonHDTestDataset(
        dataroot_path=args.data_dir,
        phase=args.phase,
        order="unpaired" if args.unpaired else "paired",
        size=(args.height, args.width),
        data_list=args.data_list,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4,
    )

    pipe = TryonPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae=vae,
            feature_extractor= CLIPImageProcessor(),
            text_encoder = text_encoder_one,
            text_encoder_2 = text_encoder_two,
            tokenizer = tokenizer_one,
            tokenizer_2 = tokenizer_two,
            scheduler = noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
    ).to(accelerator.device)
    pipe.unet_encoder = UNet_Encoder

    # pipe.enable_sequential_cpu_offload()
    # pipe.enable_model_cpu_offload()
    # pipe.enable_vae_slicing()


    img_name_counts = defaultdict(int)
    data_list = []
    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            for sample in test_dataloader:
                img_emb_list = []
                for i in range(sample['cloth'].shape[0]):
                    img_emb_list.append(sample['cloth'][i])
                
                prompt = sample["caption"]

                num_prompts = sample['cloth'].shape[0]                                        
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                if not isinstance(prompt, List):
                    prompt = [prompt] * num_prompts
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * num_prompts

                image_embeds = torch.cat(img_emb_list,dim=0)

                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                
                    prompt = sample["caption_cloth"]
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                    if not isinstance(prompt, List):
                        prompt = [prompt] * num_prompts
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * num_prompts

                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )
                    
                    generator = torch.Generator(pipe.device).manual_seed(args.seed) if args.seed is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = sample['pose_img'],
                        text_embeds_cloth=prompt_embeds_c,
                        cloth = sample["cloth_pure"].to(accelerator.device),
                        mask_image=sample['inpaint_mask'],
                        image=(sample['image']+1.0)/2.0, 
                        height=args.height,
                        width=args.width,
                        guidance_scale=args.guidance_scale,
                        ip_adapter_image = image_embeds,
                    )[0]

                for i in range(len(images)):
                    x_sample = pil_to_tensor(images[i])
                    img_name = sample['im_name'][i].split('.')[0]
                    img_name_counts[img_name] += 1
                    target_name = f"{img_name}_{img_name_counts[img_name]}.png"
                    torchvision.utils.save_image(x_sample,os.path.join(args.output_dir, target_name))
                    data_list.append({
                        "img_name": f"{args.data_dir}/{args.phase}/image/{sample['im_name'][i]}",
                        "cloth_name": f"{args.data_dir}/{args.phase}/cloth/{sample['c_name'][i]}",
                        "target_name": f"{args.output_dir}/{target_name}",
                        "prompt": sample["caption"][i],
                    })
            with open(f"{args.output_dir}/data_list.json", "w") as f:
                json.dump(data_list, f)



if __name__ == "__main__":
    main()
