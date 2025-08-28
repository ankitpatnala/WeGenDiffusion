import numpy as np
from PIL import Image
import os
from glob import glob
import random

def load_gen_arrays(path, extension, n_files):
    """
    Load n random .npy or .png files from a folder and stack them into one numpy array.

    Args:
        path (str): Directory path.
        extension (str): File extension ('.npy' or '.png').
        n_files (int): Number of files to randomly load.

    Returns:
        np.ndarray: Stacked numpy array with shape (n_files, ...).
    """
    # Collect all files with the given extension
    files = glob(os.path.join(path, f"*{extension.lower()}"))
    if not files:
        raise FileNotFoundError(f"No files with extension {extension} found in {path}")

    # Randomly sample n_files
    chosen_files = random.sample(files, min(n_files, len(files)))

    arrays = []
    for file in chosen_files:
        if extension.lower() == ".npy":
            arr = np.load(file)
        elif extension.lower() == ".png":
            img = Image.open(file).convert("L")  # grayscale
            arr = np.array(img)
            arr = np.expand_dims(arr, axis=0)  # (1, H, W) for consistency
        else:
            raise ValueError(f"Unsupported extension: {extension}")

        arrays.append(arr)
        print(f"Loaded {os.path.basename(file)} with shape {arr.shape}, dtype {arr.dtype}")

    # Stack all arrays along first dimension
    stacked = np.concatenate(arrays, axis=0)
    print(f"\nFinal stacked array shape: {stacked.shape}, dtype: {stacked.dtype}")

    return stacked


# Main
if __name__ == "__main__":
    path = "/p/project1/training2533/corradini1/WeGenDiffusion/samples"
    extension = ".npy"  # or ".png"
    n_files = 10

    gen_samples = load_gen_arrays(path, extension, n_files)
    print(gen_samples.shape)

