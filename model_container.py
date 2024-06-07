import torch

from diffusers import AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref

class ModelContainer:
    def __init__(self, args, accelerator, **kwargs):
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=torch.float16,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        self.ref_unet = UNet2DConditionModel_ref.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        ).to(accelerator.device)
        
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )

    def replace_first_conv_layer(unet_model, new_in_channels):
        # Access the first convolutional layer
        # This example assumes the first conv layer is directly an attribute of the model
        # Adjust the attribute access based on your model's structure
        original_first_conv = unet_model.conv_in
        
        if(original_first_conv == new_in_channels):
            return
        
        # Create a new Conv2d layer with the desired number of input channels
        # and the same parameters as the original layer
        new_first_conv = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            padding=1,
        )
        
        # Zero-initialize the weights of the new convolutional layer
        new_first_conv.weight.data.zero_()

        # Copy the bias from the original convolutional layer to the new layer
        new_first_conv.bias.data = original_first_conv.bias.data.clone()
        
        new_first_conv.weight.data[:, :original_first_conv.in_channels] = original_first_conv.weight.data
        
        # Replace the original first conv layer with the new one
        return new_first_conv
