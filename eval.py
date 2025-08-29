import numpy as np
#from scipy import linalg
import argparse
from prepare_test_data import load_gen_arrays
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from train import NetCDFDataset

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(real_samples, generated_samples):
    """Calculate the FID between two sets of samples.

    Args:
        real_samples (np.ndarray): Real samples, shape (N, C, H, W).
        generated_samples (np.ndarray): Generated samples, shape (M, C, H, W).

    Returns:
        float: The calculated FID score.
    """
    # Flatten the samples to 2D arrays (N, D)
    real_samples = real_samples.reshape(real_samples.shape[0], -1)
    generated_samples = generated_samples.reshape(generated_samples.shape[0], -1)

    # Calculate means and covariances
    mu1 = np.mean(real_samples, axis=0)
    sigma1 = np.cov(real_samples, rowvar=False)
    mu2 = np.mean(generated_samples, axis=0)
    sigma2 = np.cov(generated_samples, rowvar=False)

    # Calculate FID
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

def split_region(input, regions=['europe', 'tropics'], plot_slice=False):
    """
    split the dataset along latitude, North/South
    """
    input = np.flip(input, axis=1)
    output = {}
    for region in regions:
        if region == 'europe':
           # input[:, :35, 75:125], input[:, -25:, 75:125]]
            start = (0, 75)
            end = (35, 125)
        elif region == 'tropics':
            start = (33, 0)
            end = (66, input.shape[2])
        else:
            raise ValueError(f"Region {region} not recognized. Supported regions are 'europe' and 'tropics'.")
        output[region] = input[:, start[0]:end[0], start[1]:end[1]]
        if plot_slice:
            fig, ax = plt.subplots()
            ax.imshow(input[0,:,:], interpolation='nearest')
            rect = Rectangle((start[1], start[0]), end[1]-start[1], end[0]-start[0], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.savefig(f'global_{region}.png')
            fig, ax = plt.subplots()
            ax.imshow(output[region][0,:,:], interpolation='nearest')
            plt.savefig(f'slice_{region}.png')
            
    return output

def main(args):
    val_dataset = NetCDFDataset(args.val_filepath, labels=args.label)
    val_array = np.array(val_dataset.data['t2m'].values)

    # gen_array = load_gen_arrays(args.gen_filepath, n_files=args.num_samples) 

    # print(f'Successfully loaded dataset of size {gen_array.shape} and generated dataset of size {gen_array.shape}.')
    val_arrays = split_region(val_array, args.regions, True)
    exit()
    # gen_array_europe, gen_array_tropics = split_region(gen_array)
    # gen_arrays = {'europe': gen_array_europe, 'tropics': gen_array_tropics}
    unconditional_fid_values = []

    print(f'Successfully split northern and southern hemisphere.')

    if not args.conditional:
        gen_array = load_gen_arrays(args.gen_filepath, n_files=args.num_samples) 
        gen_arrays = split_region(gen_array, args.regions)

        for d in args.regions:
            print(f'Calculating FID for {d}...')
            val_array = val_arrays[d][:args.num_samples]
            gen_array = gen_arrays[d][:args.num_samples]
            #TODO: check that label is not processes
            fid_value = calculate_fid(val_array, gen_array)
            unconditional_fid_values.append(fid_value)
            print(f"FID values {fid_value} for {d} (over {args.num_samples} samples).")
        print(f"Average FID is {np.mean(unconditional_fid_values)}")
    else:
        val_labels = val_dataset.target
        #TODO: check that there is a label and extract it this is work in progress
        unique_val_labels = np.unique(val_labels)
        print(f'unique val labels are: {unique_val_labels}')
        for label in unique_val_labels:
            conditional_fid_values = []
            for d in args.regions:
                print(f"Calculating FID for {d} and label '{label}'...")
                val_array = val_arrays[d][val_labels == label]
                gen_array = load_gen_arrays(args.gen_filepath, n_files=args.num_samples, class_label=int(label))
                gen_arrays = split_region(gen_array, args.regions)
                gen_array = gen_arrays[d]
                num_samples = min(len(val_array), len(gen_array), args.num_samples)
                if num_samples < 2:
                    print(f"Skipping label '{label}' due to insufficient samples.")
                    continue
                fid_value = calculate_fid(val_array, gen_array)
                conditional_fid_values.append(fid_value)
                print(f"FID values {fid_value} for class '{label}' for {d} (over {args.num_samples} samples).")
            class_FID = np.mean(conditional_fid_values)
            unconditional_fid_values.append(class_FID)
            print(f"Average FID for class '{label}' is {class_FID}")
        print(f"Average FID (across classes) is {np.mean(unconditional_fid_values)}")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_filepath", type=str, default="", help="Path to the training data file")
    parser.add_argument("--regions", type=list, default=['europe'], help="Regions from europe and tropics")
    parser.add_argument("--label", type=str, default=None, help="label")
    parser.add_argument("--gen_filepath", type=str, default="", help="Path to the gen data file")
    parser.add_argument("--conditional", type=bool, default=False, help="Conditional Data?")
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()
    main(args)