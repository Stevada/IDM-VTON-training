#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import argparse
import functools
import gc
import logging
import math
import os
import os.path as osp
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import safetensors

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPImageProcessor, CLIPVisionModelWithProjection

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.models import ImageProjection
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel as UNet2DConditionModel_tryon
from cp_dataset import CPDatasetV2 as CPDataset, VitonHDTestDataset
from parser_args import parse_args
from diffusers.models.embeddings import get_2d_rotary_pos_embed

from hunyuan_contrainer import HunyuanContainer

from utils import combine_images_horizontally, combine_images_vertically, is_in_range

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


def save_model_card(
    repo_id: str,
    images: list = None,
    validation_prompt: str = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
    vae_path: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{base_model}** on the **{dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompt: {validation_prompt}: \n
{img_str}

Special VAE used for training: {vae_path}.
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers-training",
        "diffusers",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
   
def check_model_components(model):
    required_components = [
        'vae', 'text_encoder_one', 'text_encoder_two',
        'tokenizer_one', 'tokenizer_two', 'noise_scheduler',
        'image_encoder', 'ref_unet'
    ]
    
    missing_components = []
    for component in required_components:
        if not hasattr(model, component):
            missing_components.append(component)
    
    if missing_components:
        raise ValueError(f"Missing required model components: {', '.join(missing_components)}")
    

def check_sample_elements(sample):
    required_keys = [
        'cloth', 'caption', 'caption_cloth', 'pose_img', 
        'cloth_pure', 'inpaint_mask', 'image'
    ]
    
    missing_keys = [key for key in required_keys if key not in sample]
    
    if missing_keys:
        raise ValueError(f"Missing keys in sample: {missing_keys}")

def log_validation(unet, model, args, accelerator, weight_dtype, log_name, validation_dataloader):
    
    unet = accelerator.unwrap_model(unet)
    
    check_model_components(model)
    pipe = TryonPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae=model.vae,
            feature_extractor= CLIPImageProcessor(),
            text_encoder = model.text_encoder_one,
            text_encoder_2 = model.text_encoder_two,
            tokenizer = model.tokenizer_one,
            tokenizer_2 = model.tokenizer_two,
            scheduler = model.noise_scheduler,
            image_encoder=model.image_encoder,
            torch_dtype=weight_dtype,
    ).to(accelerator.device)
    pipe.unet_encoder = model.ref_unet
    
        # Extract the images
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            image_logs = []
            for sample in validation_dataloader:
                check_sample_elements(sample)
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


                    with torch.inference_mode():
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
                    
                    
                    inference_guidance_scale = [0.99, 2, 5]
                    output_images = []
                    
                    target_size = sample['image'].shape[2:]
                    sample['pose_img'] = F.interpolate(sample['pose_img'], size=target_size, mode='bilinear', align_corners=False)
                    sample['cloth_pure'] = F.interpolate(sample['cloth_pure'], size=target_size, mode='bilinear', align_corners=False)


                    for scale in inference_guidance_scale:
                        generator = torch.Generator(pipe.device).manual_seed(args.seed) if args.seed is not None else None
                        (images, inference_sampling_images, before_inference_images) = pipe(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                            num_inference_steps=args.inference_steps,
                            generator=generator,
                            strength=0.8,
                            pose_img=sample['pose_img'],
                            text_embeds_cloth=prompt_embeds_c,
                            cloth=sample['cloth_pure'].to(accelerator.device),
                            mask_image=sample['inpaint_mask'],
                            image=(sample['image'] + 1.0) / 2.0,
                            height=args.height,
                            width=args.width,
                            guidance_scale=scale,
                            ip_adapter_image=image_embeds,
                            inference_sampling_step=args.inference_sampling_step,
                        )
                        
                        # print(f"image: {images}, internal_images: {inference_sampling_images}")
                        # os._exit(os.EX_OK)
                        image = images[0]
                        internal_images = []
                        for before_var in before_inference_images:
                            internal_images.append(before_var[0])
                        for internal_var in inference_sampling_images:
                            internal_images.append(internal_var[0])
                        internal_images.append(image)
                        horizontal_image = combine_images_horizontally(internal_images)
                        output_images.append(horizontal_image)

                    # Combine the images into one
                    combined_image = combine_images_vertically(output_images)

                    image_logs.append({
                        "garment": sample["cloth_pure"], 
                        "model": sample['image'], 
                        "orig_img": sample['image'], 
                        "samples": combined_image, 
                        "prompt": prompt,
                        "inpaint mask": sample['inpaint_mask'],
                        "pose_img": sample['pose_img'],
                        })
                        
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    if not is_wandb_available():
                        raise ImportError("Make sure to install wandb if you want to use it for logging during validation.")
                    import wandb
                    formatted_images = []
                    for log in image_logs:
                        # logger.info("Adding image to tacker")
                        formatted_images.append(wandb.Image(log["garment"], caption="garment images"))
                        # formatted_images.append(wandb.Image(log["model"], caption="masked model images"))
                        formatted_images.append(wandb.Image(log["orig_img"], caption="original images"))
                        formatted_images.append(wandb.Image(log["inpaint mask"], caption="inpaint mask"))
                        formatted_images.append(wandb.Image(log["pose_img"], caption="pose_img"))
                        formatted_images.append(wandb.Image(log["samples"], caption=log["prompt"]))
                    tracker.log({log_name: formatted_images})
                else:
                    logger.warn(f"image logging not implemented for {tracker.name}")            

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
import torch
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)

def encode_prompt(
    prompt: Union[str, List[str]],
    tokenizer,
    text_encoder,
    device: torch.device = None,
    dtype: torch.dtype = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    max_sequence_length: Optional[int] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`):
            prompt or list of prompts to be encoded
        tokenizer: Tokenizer to use for encoding the prompt
        text_encoder: Text encoder to use for encoding the prompt
        device: (`torch.device`):
            torch device
        dtype (`torch.dtype`):
            torch dtype
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        prompt_attention_mask (`torch.Tensor`, *optional*):
            Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
        max_sequence_length (`int`, *optional*): maximum sequence length to use for the prompt.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if max_sequence_length is None:
        max_length = tokenizer.model_max_length
    else:
        max_length = max_sequence_length

    if prompt_embeds is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            attention_mask=prompt_attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # duplicate attention mask for each generation per prompt
    if prompt_attention_mask is not None:
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

    return prompt_embeds, prompt_attention_mask

import torch
from typing import Union

def get_resize_crop_region_for_grid(src, tgt_size):
    th = tw = tgt_size
    h, w = src

    r = h / w

    # resize
    if r > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)



def prepare_image(
    image: Union[torch.Tensor, List[torch.Tensor]],
    width: int,
    height: int,
    batch_size: int,
    num_images_per_prompt: int,
    device: torch.device,
    dtype: torch.dtype,
    image_processor = None,
):
    if isinstance(image, torch.Tensor):
        pass
    else:
        if image_processor is None:
            raise ValueError("image_processor must be provided if image is not a torch.Tensor")
        image = image_processor.preprocess(image, height=height, width=width)

    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    return image

def encode_image(model, image, device, num_images_per_prompt, output_hidden_states=None):
    dtype = next(model.image_encoder.parameters()).dtype
    if not isinstance(image, torch.Tensor):
        image = model.feature_extractor(image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    # print(f"encode image (initial): {image.dtype}")
    if output_hidden_states:
        image_enc_hidden_states = model.image_encoder(image, output_hidden_states=True).hidden_states[-2]
        # print(f"encode image (after encoding): {image_enc_hidden_states.dtype}")
        
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        # print(f"encode image (after repeat_interleave): {image_enc_hidden_states.dtype}")
        
        uncond_image_enc_hidden_states = model.image_encoder(
            torch.zeros_like(image), output_hidden_states=True
        ).hidden_states[-2]
        # print(f"encode image (uncond encoding): {uncond_image_enc_hidden_states.dtype}")
        
        uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        # print(f"encode image (uncond after repeat_interleave): {uncond_image_enc_hidden_states.dtype}")

        # print(f"encode image (final): {image_enc_hidden_states.dtype}")


        return image_enc_hidden_states, uncond_image_enc_hidden_states
    else:
        image_embeds = model.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds


def compute_vae_encodings(pixel_values, vae):
    
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights

def load_model_with_zeroed_mismatched_keys(unet, pretrained_weights_path):
    # Load the pretrained weights
    # new weight as an initialized state
    # Determine the file type and load the pretrained weights
    if pretrained_weights_path.endswith('.safetensors'):
        state_dict = safetensors.torch.load_file(pretrained_weights_path)
    elif pretrained_weights_path.endswith('.bin'):
        state_dict = torch.load(pretrained_weights_path)
    else:
        raise ValueError("Unsupported file type. Only .safetensors and .bin are supported.")
    
    # Initialize a new state dict for the model
    # The new unet data structure
    new_state_dict = unet.state_dict()
    
    # Iterate through the pretrained weights
    for key, value in state_dict.items():
        if key in new_state_dict and new_state_dict[key].shape == value.shape:
            new_state_dict[key] = value
        else:
            if key in unet.attn_processors.keys():
                # Initialize the ip adaptor for the model check https://github.com/tencent-ailab/IP-Adapter/blob/cfcdf8ce36f31e3d358b3c4c4b1bb78eab2854bd/tutorial_train_plus.py#L335
                layer_name = key.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet[layer_name + ".to_v.weight"],
                }
                new_state_dict[key] = weights
            else:
                print(f"Key {key} mismatched or not found in model. Initializing with zeros.")
                new_state_dict[key] = torch.zeros_like(new_state_dict[key])
    
    # Load the new state dict into the model
    unet.load_state_dict(new_state_dict)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
        if args.tracker_project_name:
            tracker_config = dict(vars(args))
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs={"wandb": {"entity": args.tracker_entity}})


    test_dataset = VitonHDTestDataset(
        dataroot_path=args.dataroot,
        phase="test",
        order="paired",
        size=(args.height, args.width),
        data_list=args.validation_data_list,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    model = HunyuanContainer(args,accelerator)
    
    # log_validation( model, args, accelerator, torch.float16, "old_model", test_dataloader)

    # Freeze all components
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.text_encoder_2.requires_grad_(False)
    model.transformer.requires_grad_(False)

    # If there's a controlnet in the Hunyuan model, freeze it
    if hasattr(model, 'controlnet'):
        model.controlnet.train()

    # os._exit(os.EX_OK)
    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move components to device and cast to appropriate dtype
    # The VAE is kept in float32 to avoid NaN losses
    model.vae.to(accelerator.device, dtype=torch.float32)

    # Move text encoders to device and cast to weight_dtype
    model.text_encoder.to(accelerator.device, dtype=weight_dtype)
    model.text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # Move transformer (main model) to device and cast to weight_dtype
    model.transformer.to(accelerator.device, dtype=weight_dtype)
    
    if hasattr(model, 'controlnet'):
        model.controlnet.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if args.gradient_checkpointing:
        model.controlnet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = model.controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # os._exit(os.EX_OK)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataroot is None:
        assert "Please provide correct data root"
    # train_dataset = CPDataset(args.dataroot, args.resolution, mode="train", data_list=args.train_data_list)
    validation_dataset = VitonHDTestDataset(
        dataroot_path=args.dataroot,
        phase="test",
        order="paired",
        size=(args.height, args.width),
        data_list=args.validation_data_list,
    )
    
    train_dataset = VitonHDTestDataset(
        dataroot_path=args.dataroot,
        phase="train",
        order="paired",
        size=(args.height, args.width),
        data_list=args.train_data_list,
    )
    
    # os._exit(os.EX_OK)
    
    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory. We will pre-compute the VAE encodings too.
    text_encoders = [model.text_encoder, model.text_encoder_2]
    tokenizers = [model.tokenizer, model.tokenizer_2]
    
    gc.collect()
    torch.cuda.empty_cache()

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # os._exit(os.EX_OK)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model.transformer, model.controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model.transformer, model.controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune-sdxl", config=vars(args))

    # Function for unwrapping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if torch.backends.mps.is_available() or "playground" in args.pretrained_model_name_or_path:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # put untrain modules into freeze

    # developing log
    # print(f"the in channel number: {unet.config.in_channels}")
    # log_validation(unet, model, args, accelerator, weight_dtype, "pre_train", validation_dataloader)
    
    # os._exit(os.EX_OK)
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # generator = torch.Generator(accelerator.device).manual_seed(args.seed) if args.seed is not None else None


    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model.controlnet):
                check_sample_elements(batch)
                
                prompt = batch["caption"]
                target_size = batch['image'].shape[2:]
                cloth = batch["cloth_pure"]

                num_prompts = batch['cloth'].shape[0]    
                
                width=args.width
                height=args.height                                
                
                # print(f"image_embeds type {image_embeds.dtype}")
                # os._exit(os.EX_OK)
                   
                (
                    prompt_embeds,
                    prompt_attention_mask,
                ) = encode_prompt(
                    prompt=prompt,
                    tokenizer=model.tokenizer,
                    text_encoder=model.text_encoder,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    num_images_per_prompt=1,
                )
                (
                    prompt_embeds_2,
                    prompt_attention_mask_2,
                ) = encode_prompt(
                    prompt=prompt,
                    tokenizer=model.tokenizer_2,
                    text_encoder=model.text_encoder_2,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    num_images_per_prompt=1,
                )
                
                control_image = prepare_image(
                    image=cloth,
                    width=args.width,
                    height=args.height,
                    batch_size=args.train_batch_size,
                    num_images_per_prompt=1,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    image_processor=model.image_processor
                )
                height, width = control_image.shape[-2:]

                control_image = model.vae.encode(control_image).latent_dist.sample()
                control_image = control_image * model.vae.config.scaling_factor

                
                # print(f"prompt_embeds shape: {prompt_embeds.shape}")
                # os._exit(os.EX_OK)
                
                # is_0_1 = is_in_range(batch["image"], 0 , 1)
                # print(f"is_0_1: {is_0_1}")
                init_image = (batch["image"] + 1.0) / 2.0
                init_image = model.image_processor.preprocess(
                    init_image, height=args.height, width=args.width, crops_coords=None, resize_mode="default"
                )
                
                
                input_latent = compute_vae_encodings(init_image, model.vae)
                # print(f"model_input: {model_input.shape}")
                                
                cloth_latents = compute_vae_encodings(cloth, model.vae)
                # print(f"cloth_latents: {cloth_latents.shape}")
                # os._exit(os.EX_OK)
                reconstruct_model_input = model.reconstruct_vae_img(input_latent)
                reconstruct_cloth_image = model.reconstruct_vae_img(cloth_latents)
                # print(f"reconstruct_model_input, {reconstruct_model_input}")
                # print(f"reconstruct_pose_image, {reconstruct_pose_image}")
                # print(f"reconstruct_cloth_image, {reconstruct_cloth_image}")

                # formatted_images = []
                # formatted_images.append(wandb.Image(reconstruct_model_input[0], caption="reconstruct_model_input"))
                # formatted_images.append(wandb.Image(reconstruct_pose_image[0], caption="reconstruct_pose_image"))
                # formatted_images.append(wandb.Image(reconstruct_cloth_image[0], caption="reconstruct_cloth_image"))
                # accelerator.log({"trainning internal image input": formatted_images})
                
                noise = torch.randn_like(input_latent)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (input_latent.shape[0], input_latent.shape[1], 1, 1), device=accelerator.device
                    )

                bsz = input_latent.shape[0]
                if args.timestep_bias_strategy == "none":
                    # Sample a random timestep for each image without bias.
                    timesteps = torch.randint(
                        0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
                    )
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = generate_timestep_weights(args, model.noise_scheduler.config.num_train_timesteps).to(
                        input_latent.device
                    )
                    timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                # print(f"timesteps of training: {timesteps}")
                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # Sample noise that we'll add to the latents

                noisy_model_input = model.noise_scheduler.add_noise(input_latent, noise, timesteps)
                # print(f"noisy_model_input shape: {noisy_model_input.shape}")
                # os._exit(os.EX_OK)
                
                latent_model_input = noisy_model_input
                       
                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.height, args.width)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids
                
                original_size = (args.height, args.width)
                crops_coords_top_left = (0, 0)
                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip([original_size] * args.train_batch_size, [crops_coords_top_left] * args.train_batch_size)]
                )
                
                # print(f"add_text_embeds shape: {add_text_embeds.shape}")
                # print(f"add_time_ids shape: {add_time_ids.shape}")
                
                t_expand = torch.tensor([timesteps] * latent_model_input.shape[0], device=accelerator.device).to(
                    dtype=latent_model_input.dtype
                )
                
                # 8. create image_rotary_emb, style embedding & time ids
                grid_height = height // 8 // model.transformer.config.patch_size
                grid_width = width // 8 // model.transformer.config.patch_size
                base_size = 512 // 8 // model.transformer.config.patch_size
                grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
                image_rotary_emb = get_2d_rotary_pos_embed(
                    model.transformer.inner_dim // model.transformer.num_heads, grid_crops_coords, (grid_height, grid_width)
                )
                
                style = torch.tensor([0], device=accelerator.device)
                style = style.to(device=accelerator.device).repeat(args.train_batch_size * 1)

                control_block_samples = model.controlnet(
                    latent_model_input,
                    t_expand,
                    encoder_hidden_states=prompt_embeds,
                    text_embedding_mask=prompt_attention_mask,
                    encoder_hidden_states_t5=prompt_embeds_2,
                    text_embedding_mask_t5=prompt_attention_mask_2,
                    image_meta_size=add_time_ids,
                    style=style,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    controlnet_cond=control_image,
                    conditioning_scale=1.0,
                )[0]
                
                # print(f"reference_features: {reference_features}")
                # os._exit(os.EX_OK)
                                
                model_pred = model.transformer(
                    latent_model_input,
                    t_expand,
                    encoder_hidden_states=prompt_embeds,
                    text_embedding_mask=prompt_attention_mask,
                    encoder_hidden_states_t5=prompt_embeds_2,
                    text_embedding_mask_t5=prompt_attention_mask_2,
                    image_meta_size=add_time_ids,
                    style=style,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    controlnet_block_samples=control_block_samples,
                )[0]
                
                model_pred, _ = model_pred.chunk(2, dim=1)

                
                # # latents = model.noise_scheduler.step(model_pred, timesteps, latent_model_input, return_dict=False)[0]
                # reconstruct_latent_image = model.reconstruct_vae_img(model_pred)
                # # print(f"reconstruct_latent_image: {reconstruct_latent_image}")
                # formatted_images = []
                # formatted_images.append(wandb.Image(reconstruct_latent_image[0], caption="reconstruct_latent_image"))
                # accelerator.log({"trainning internal image output": formatted_images})
                
                # os._exit(os.EX_OK)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    model.noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if model.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif model.noise_scheduler.config.prediction_type == "v_prediction":
                    target = model.noise_scheduler.get_velocity(input_latent, noise, timesteps)
                elif model.noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = input_latent
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {model.noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(model.noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if model.noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif model.noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # if global_step % args.validation_steps == 0:
                        # log_validation(unet, model, args, accelerator, weight_dtype, "during_train", validation_dataloader)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = unwrap_model(unet)
    #     if args.use_ema:
    #         ema_unet.copy_to(unet.parameters())

    #     # Serialize pipeline.
    #     vae = AutoencoderKL.from_pretrained(
    #         vae_path,
    #         subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
    #         revision=args.revision,
    #         variant=args.variant,
    #         torch_dtype=weight_dtype,
    #     )
    #     pipeline = StableDiffusionXLPipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         unet=unet,
    #         vae=vae,
    #         revision=args.revision,
    #         variant=args.variant,
    #         torch_dtype=weight_dtype,
    #     )
    #     if args.prediction_type is not None:
    #         scheduler_args = {"prediction_type": args.prediction_type}
    #         pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    #     pipeline.save_pretrained(args.output_dir)

    #     # run inference
    #     images = []
    #     if args.validation_prompt and args.num_validation_images > 0:
    #         pipeline = pipeline.to(accelerator.device)
    #         generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    #         with autocast_ctx:
    #             images = [
    #                 pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
    #                 for _ in range(args.num_validation_images)
    #             ]

    #         for tracker in accelerator.trackers:
    #             if tracker.name == "tensorboard":
    #                 np_images = np.stack([np.asarray(img) for img in images])
    #                 tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
    #             if tracker.name == "wandb":
    #                 tracker.log(
    #                     {
    #                         "test": [
    #                             wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
    #                             for i, image in enumerate(images)
    #                         ]
    #                     }
    #                 )

    #     if args.push_to_hub:
    #         save_model_card(
    #             repo_id=repo_id,
    #             images=images,
    #             validation_prompt=args.validation_prompt,
    #             base_model=args.pretrained_model_name_or_path,
    #             dataset_name=args.dataset_name,
    #             repo_folder=args.output_dir,
    #             vae_path=args.pretrained_vae_model_name_or_path,
    #         )
    #         upload_folder(
    #             repo_id=repo_id,
    #             folder_path=args.output_dir,
    #             commit_message="End of training",
    #             ignore_patterns=["step_*", "epoch_*"],
    #         )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
