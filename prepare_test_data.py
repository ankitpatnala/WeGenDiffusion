import numpy as np
from PIL import Image
import os
from glob import glob
import random


def load_gen_arrays(path, extension, n_files, class_label=None):
    """
    Load n random .npy or .png files from a folder (optionally inside a class label subfolder) 
    and stack them into one numpy array (N, H, W).

    Args:
        path (str): Directory path.
        extension (str): File extension ('.npy' or '.png').
        n_files (int): Number of files to randomly load.
        class_label (str or int, optional): If provided, load from subfolder named as the class label.

    Returns:
        np.ndarray: Stacked numpy array with shape (n_files, H, W).
    """
    if class_label is not None:
        path = os.path.join(path, str(class_label))
    
    files = glob(os.path.join(path, f"*{extension.lower()}"))
    if not files:
        raise FileNotFoundError(f"No files with extension {extension} found in {path}")

    chosen_files = random.sample(files, min(n_files, len(files)))

    arrays = []
    for file in chosen_files:
        if extension.lower() == ".npy":
            arr = np.load(file)
        elif extension.lower() == ".png":
            img = Image.open(file).convert("L")  # grayscale
            arr = np.array(img)
        else:
            raise ValueError(f"Unsupported extension: {extension}")

        # Remove singleton first dimension if it exists (C=1)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)

        arrays.append(arr)
        print(f"Loaded {os.path.basename(file)} with shape {arr.shape}, dtype {arr.dtype}")

    stacked = np.stack(arrays, axis=0)
    print(f"\nFinal stacked array shape: {stacked.shape}, dtype: {stacked.dtype}")

    return stacked


# Main
if __name__ == "__main__":
    path = "/p/project1/training2533/corradini1/WeGenDiffusion/samples-months"
    extension = ".npy"  # or ".png"
    n_files = 10
    class_label = 0

    gen_samples = load_gen_arrays(path, extension, n_files, class_label=class_label)
    print(gen_samples.shape)

