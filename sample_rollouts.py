"""
Sample Rollouts Script

This script performs rollouts using a trained DiT model on ERA5 2-degree temperature data.
It loads a trained model checkpoint, prepares input data, and generates samples using a diffusion process.
The generated samples are saved as .npy files for further analysis.
"""
import xarray as xr
import torch
import matplotlib.pyplot as plt
from models import DiT_models
from diffusion import create_diffusion
import numpy as np

# ------------------------------------------------------------------------------
# Data and Model Configuration
# ------------------------------------------------------------------------------

# Path to the ERA5 2-degree temperature NetCDF dataset
data_path = "/p/project1/training2533/lancelin1/WeGenDiffusion/data/2011_t2m_era5_2deg.nc"
# Path to the trained model checkpoint
model_name = "DiT-B-2_previous_state_1"
model_path = f"/p/project1/training2533/lancelin1/WeGenDiffusion/results/{model_name}/ckpt_0000110.pt"
# Number of rollouts to perform (i.e., how many times to sample the diffusion process)
num_rollouts = 28
# Indices of the time steps to use as initial conditions for rollouts
indices = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] # this corresponds to different initial conditions # (we will compute ACC by avering over them)
num_indices = len(indices)
# Shape of the input/output tensors: (batch, channels, height, width)
shape = (num_indices, 1, 90, 180)

# Output file naming configuration
output_name = model_name
labels = "previous_state"     # Label type for the model
diffusion_steps = 500 # number of denoising steps, carreful: choose the same during the training (mostly 500)
combine_imgs = False          # Not used in this script, but kept for clarity
vars = ['t2m']                # List of variables to use (here, only temperature at 2m)

# ------------------------------------------------------------------------------
# Data Loading and Preprocessing
# ------------------------------------------------------------------------------
# Load the ERA5 dataset using xarray
data = xr.open_dataset(data_path)
# Compute the mean and standard deviation for normalization
mean = data.mean()
std = data.std()
# Prepare the input tensor 'y' by normalizing the selected indices for the specified variable(s)
# The resulting shape will be (num_indices, 1, 90, 180)
y = np.concatenate([
    np.expand_dims(
        (data.isel(valid_time=index)[var].values - mean[var].values) / std[var].values,
        axis=0
    )
    for var in vars
    for index in indices
])
# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
# Load the model state dictionary from the checkpoint
model_state_dict = torch.load(model_path, map_location='cuda')['model']
# Instantiate the DiT model with the appropriate configuration
model = DiT_models['DiT-B/2'](
    input_size=(90, 180),
    num_classes=1000,
    labels=labels
).to('cuda')
# Load the model weights
model.load_state_dict(model_state_dict, strict=True)
# Set the model to evaluation mode (disables dropout, etc.)
model.eval()

# ------------------------------------------------------------------------------
# Rollout Loop: Generate and Save Samples
# ------------------------------------------------------------------------------

for i in range(1, num_rollouts + 1):
    print(f"Rollout {i} of {num_rollouts}")
    # Create a new diffusion process for each rollout
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=100)
    print("Input y shape before tensor conversion:", y.shape)
    y_tensor = torch.tensor(y, dtype=torch.float32, device='cuda')
    # Reshape the tensor to match the expected input shape for the model
    y_tensor = y_tensor.reshape(num_indices, 1, 90, 180)
    print("Input y shape after reshaping:", y_tensor.shape)
    # Generate denoised images using the diffusion model
    # The model_kwargs argument provides the conditional input (y_tensor)
    denoised_images = diffusion.p_sample_loop(
        model,
        shape,
        model_kwargs=dict(y=y_tensor),
        progress=True
    )
    # For the next rollout, use the generated images as the new input
    y = denoised_images.detach().clone()
    # Move the generated images to CPU and convert to numpy array for saving
    denoised_images_np = denoised_images.cpu().numpy()
    # Save the generated images to a .npy file for later analysis
    np.save(f'samples/forecast/{output_name}_rollout_{i}.npy', denoised_images_np)
# End of script
