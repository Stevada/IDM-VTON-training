import torch
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import BertModel, BertTokenizer, T5EncoderModel, MT5Tokenizer
from diffusers.models import HunyuanDiT2DModel, HunyuanDiT2DControlNetModel, HunyuanDiT2DMultiControlNetModel

class HunyuanContainer:
    def __init__(self, args, accelerator, **kwargs):
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
        )
        self.transformer = HunyuanDiT2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.float16,
        ).to(accelerator.device)
        
        self.text_encoder = BertModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )
        self.tokenizer_2 = MT5Tokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            use_fast=False,
        )
        
        self.scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        
        self.controlnet = HunyuanDiT2DControlNetModel.from_pretrained(
            args.controlnet_model_name_or_path,
            torch_dtype=torch.float16,
        ).to(accelerator.device)
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    
