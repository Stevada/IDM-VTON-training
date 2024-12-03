#VITON-HD
##paired setting
accelerate launch img_generator/generate.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "IDM_test" --data_dir "/workspace/VITON-HD" \
    --seed 42 --test_batch_size 1 --guidance_scale 2.0 \
    --data_list "generate_data_list.txt" --phase "test" \
    --unpaired


# ##unpaired setting
# accelerate launch inference.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/Dataset/zalando" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0



# #DressCode
# ##upper_body
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "upper_body"

# ##lower_body
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "lower_body"

# ##dresses
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "dresses"
