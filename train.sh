export MODEL_NAME="./checkpoints/IDM-VTON"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
export GARMENT_NAME="stabilityai/stable-diffusion-xl-base-1.0"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_garment_model_name_or_path=$GARMENT_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512  \
  --center_crop  \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --checkpointing_steps=5000 \
  --output_dir="output" \
  --dataroot="./data/VITON-HD"  \
  --train_data_list="subtrain_1.txt" \
  --validation_data_list="subtrain_1.txt" \
  --original_model_name_or_path="./checkpoints/IDM-VTON"  \
  --image_encoder_path="./checkpoints/image_encoder"  \
  --width=512   \
  --height=512  \
  # --enable_xformers_memory_efficient_attention \
