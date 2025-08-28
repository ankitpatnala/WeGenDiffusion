import xarray as xr
import torch
import matplotlib.pyplot as plt
import torch
from models import DiT_models
from diffusion import create_diffusion

model_path = "./results-month/DiT-B-2/ckpt_0000100.pt"
data_path = "/p/project1/training2533/jendersie1/WeGenDiffusion/data/2011_t2m_era5_2deg.nc"
num_labels = 12
shape = (num_labels,1,90,180)
generate = True
output_name = 'month'
axis_name = 'month'
combine_imgs = False

if generate:
    model_state_dict = torch.load(model_path, map_location='cuda')['model']
    model = DiT_models['DiT-B/2'](input_size=(90,180),num_classes=1000).to('cuda')
    model.load_state_dict(model_state_dict,strict=True)
    model.eval()

    diffusion = create_diffusion(timestep_respacing="")
    y = torch.arange(0, shape[0],dtype=torch.long,device='cuda')

    denoised_images = diffusion.p_sample_loop(model,shape,model_kwargs=dict(y=y))
    # save the raw outputs since generation is expensive
    torch.save(denoised_images, f'{output_name}_samples.pt')

else:
    denoised_images = torch.load(f'{output_name}_samples.pt')

denoised_images = torch.squeeze(denoised_images, 1)

ds = xr.open_dataset(data_path)
mean = ds.mean()['t2m'].values
std = ds.std()['t2m'].values

samples = []
for i in range(shape[0]):
    samples.append(xr.Dataset({
    "t2m": (("lat", "lon"), mean + std*denoised_images[i].cpu().numpy())
    },
    coords={
        'lon':ds['lon'],
         'lat':ds['lat']}))

combined = xr.concat(samples,dim=axis_name)

combined["t2m"].plot(
    col=axis_name,
    col_wrap=6,
    cmap="viridis",
    cbar_kwargs={
        "orientation": "vertical",   # vertical colorbar at the side
        "pad": 0.05,                  # distance from the plots
        "shrink": 0.8                  # shrink length
    },
    vmin=220,
    vmax=320
)
plt.savefig(f'{output_name}_samples.png')