import os 
import random

data_dir = "/home/stevexu/data/VITON-HD/test"
model_folder = "image"
cloth_folder = "cloth"

# Get list of all cloth and model files
cloth_files = os.listdir(os.path.join(data_dir, cloth_folder))
model_files = os.listdir(os.path.join(data_dir, model_folder))

# Create a list to store the paired data
paired_data = []

# Generate Cartesian product of selected models and clothes
for model_file in model_files:
    # Ensure the cloth file is different from the model file
    cloth_file = model_file 
    while cloth_file == model_file:
        cloth_file = random.choice(cloth_files)
    paired_data.append(f"{model_file} {cloth_file}")

# Write the paired data to a file
output_file = "/workspace/VITON-HD/generate_data_list.txt"
with open(output_file, "w") as f:
    for pair in paired_data:
        f.write(f"{pair}\n")

print(f"Generated {len(paired_data)} paired data entries in {output_file}")
