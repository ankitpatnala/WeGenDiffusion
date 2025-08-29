
import numpy as np
from train import NetCDFDataset 
import xarray as xr
import matplotlib.pyplot as plt
import argparse
from prepare_test_data import load_gen_arrays

#from eval import calculate_fid

def split_region(input):
    """
    split the dataset along latitude, North/South
    """
    output1=input[:,0:45,:]
    output2=input[:,45:90,:]
    return output1, output2
    # 
def ACC(sample_dataset,reference_dataset,mean_dataset):
    residual_dataset=sample_dataset-mean_dataset
    residual_reference=reference_dataset-mean_dataset
    ACC_res=np.corrcoef(residual_dataset.flatten(),residual_reference.flatten())[0,1]
    return ACC_res

 
def zonal_averaged_power_spectrum_mod(field, time_avg=True):
    """
    This function calculates the zonal averaged power spectrum of a given field. It is designed to work with 
    numpy array (time,lat,lon)
    
    Parameters:
    - 
    Returns:
    - power_spectrum_avg (xarray.DataArray): The zonal averaged power spectrum of the input field.

   
    """
    
    n_lon =field.shape[2]

    ###########################################################################################
    field_fft = np.fft.rfft(field, axis=2, norm='forward') # Convention used: the first Fourier coefficient is the mean of the field

    # Compute the power spectrum (squared magnitude of Fourier coefficients)
    power_spectrum = np.abs(field_fft)**2

    # Define the zonal wavenumbers
    nx = n_lon
    #print("n_x =" , nx)
    k_x = np.fft.fftfreq(nx, d=1/nx)
   

    # Only take the positive frequencies (or the first half if using real FFT)
    k_x = k_x[:nx//2]
    power_spectrum = power_spectrum[:nx//2]
    # count the positive frequencies twice except for the first one (zero frequency), because the FFT of a real function is symmetric
    power_spectrum[1:] *= 2
    # multiply by a factor cos(pi latitude[i] / 180) in axis 1
    # C0 = 40.075*10**6 # Earth's circumference in meters
    #weights = np.cos(np.pi * coords['lat']/180) # * C0
    #weights = (weights / weights.mean())
    #weights = weights.reshape(1, -1, *([1] * (power_spectrum.ndim - 2)))
    #power_spectrum *= weights 
    # Average the power spectrum over latitudes and time (axis 1 and 2)
    
   
    print("Warning: 'time' dimension detected. Averaging over the time dimension.")
    power_spectrum_avg = power_spectrum.mean(axis=( 0,1))

       # print(len(initial_field.coords))
    print("Shape after averaged FFT: ", power_spectrum_avg.shape)
    ################################################################################################

    return k_x,  power_spectrum_avg   

def lag_linregress_3D(x, y, lagx=0, lagy=0):
    """
    Input: Two xr.Datarrays of any dimensions with the first dim being time. 
    Thus the input data could be a 1D time series, or for example, have three 
    dimensions (time,lat,lon). 
    Datasets can be provided in any order, but note that the regression slope 
    and intercept will be calculated for y with respect to x.
    Output: Covariance, correlation, regression slope and intercept, p-value, 
    and standard error on regression between the two datasets along their 
    aligned time dimension.  
    Lag values can be assigned to either of the data, with lagx shifting x, and
    lagy shifting y, with the specified lag amount. 
    """ 
    #1. Ensure that the data are properly alinged to each other. 
    x,y = xr.align(x,y)
    
    #2. Add lag information if any, and shift the data accordingly
    if lagx!=0:
    
        # If x lags y by 1, x must be shifted 1 step backwards. 
        # But as the 'zero-th' value is nonexistant, xr assigns it as invalid 
        # (nan). Hence it needs to be dropped
        x   = x.shift(time = -lagx).dropna(dim='time')
    
        # Next important step is to re-align the two datasets so that y adjusts
        # to the changed coordinates of x
        x,y = xr.align(x,y)
    
    if lagy!=0:
        y   = y.shift(time = -lagy).dropna(dim='time')
        x,y = xr.align(x,y)
    
    #3. Compute data length, mean and standard deviation along time axis: 
    n = y.notnull().sum(dim='time')
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xstd  = x.std(axis=0)
    ystd  = y.std(axis=0)
    
    #4. Compute covariance along time axis
    cov   =  np.sum((x - xmean)*(y - ymean), axis=0)/(n)
    
    #5. Compute correlation along time axis
    cor   = cov/(xstd*ystd)
    
    #6. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope  
    
    #7. Compute P-value and standard error
    #Compute t-statistics
    tstats = cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
    stderr = slope/tstats
    
    #8 rmse
    rmse=(np.sum((x - y)*(x - y), axis=0)/(n))**.5
    
    from scipy.stats import t
    pval   = t.sf(tstats, n-2)*2
    pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)
    
    return cov,cor,slope,intercept,pval,stderr,rmse   

def map_rmse_cor(dataset,sample,fileo):# map sif
    res=lag_linregress_3D(dataset,sample)
    rmse=res[6]
    cor=res[1]
    # map of monthly mean
    import cartopy
    import cartopy.crs as ccrs
    from importlib.machinery import SourceFileLoader
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    mc = SourceFileLoader("my_colors", "/perm/cxsg/repos/Module/From_Pete/Packages/my_colors.py").load_module()


    temp=mc.temp
    
    # Check performance on a map to see spatial performance of XGBoost trained model
    projection = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(nrows=2,ncols=1, figsize=[18,12], subplot_kw={'projection': projection})

    for axes in ax.flatten():
            axes.coastlines()
            gl = axes.gridlines(draw_labels=True, alpha=.3, color='grey', 
                              linewidth=0.5,linestyle ='-', 
                              x_inline= False, y_inline=False)
            axes.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
            gl.n_steps = 100
            gl.top_labels = False
            gl.right_labels = False
        
   
    
   
    
    h=rmse.plot.pcolormesh(ax=ax[0], transform=ccrs.PlateCarree(),vmin=0,vmax=2,add_colorbar=False,cmap='jet')
    ax[0].set_title('temporal RMSD between dataset and generated samples',fontsize=25)
    
    # h=stderr.isel(time=itime[2]).plot.pcolormesh(ax=ax[1][0], transform=ccrs.PlateCarree(),add_colorbar=False,cmap=vegdiff)
    # ax[1][0].set_title(targetname+ ' increment  '+itime_name[2],fontsize='x-large')
    
   
    #cbar_ax = fig.add_axes([0.2, 0.45, 0.6, 0.05]) 
    #divider = make_axes_locatable(ax[0])
    #cbar_ax = divider.new_vertical(size = "5%",
     #                      pad = 0.3,
     #                      pack_start = True)
    cbar = fig.colorbar(h,extend='both', orientation='vertical')
    cbar.set_label(' RMSD',fontsize=25,rotation='vertical')
    cbar.ax.tick_params(labelsize=25)
    
    
    h=cor.plot.pcolormesh(ax=ax[1], transform=ccrs.PlateCarree(),vmin=0,vmax=1,add_colorbar=False,cmap='jet')
    ax[1].set_title('temporal correlation between dataset and generated samples',fontsize=25)
    
    # h=stderr.isel(time=itime[2]).plot.pcolormesh(ax=ax[1][0], transform=ccrs.PlateCarree(),add_colorbar=False,cmap=vegdiff)
    # ax[1][0].set_title(targetname+ ' increment  '+itime_name[2],fontsize='x-large')
    
   
    #cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.05]) 
    #cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.05]) 
    #divider = make_axes_locatable(ax[1])
    #cbar_ax = divider.new_vertical(size = "5%",
    #                       pad = 0.3,
    #                      pack_start = True)
    cbar = fig.colorbar(h,extend='both', orientation='vertical')
    cbar.set_label(' correlation',fontsize=25,rotation='vertical')
    cbar.ax.tick_params(labelsize=25)
    #cbar.set_label(' correlation',fontsize='x-large',rotation='horizontal')

	#fig.tight_layout()
    plt.savefig(fileo)   

def main(args):
    print("new eval power spectra")
    

    #load dataset and genrated samples and transform into nparray
    #train_dataset = NetCDFDataset(args.train_filepath)
    #train_array = np.array(train_dataset.data['t2m'].values)[:args.num_samples]
    if args.expid=="prevtimestep":
        train_array = load_gen_arrays(args.train_filepath, n_files=args.num_samples) 
        gen_array = load_gen_arrays(args.gen_filepath, n_files=args.num_samples) 
    else:
        train_dataset = NetCDFDataset(args.train_filepath)
        train_array = np.array(train_dataset.data['t2m'].values)[:args.num_samples]
        gen_array = load_gen_arrays(args.gen_filepath, n_files=args.num_samples) 


    #gen_dataset=np.copy(train_dataset)
    
   # gen_array = load_gen_arrays(args.gen_filepath, n_files=args.num_samples) 
    #gen_dataset['t2m'].values=gen_array
    print(f'Successfully loaded dataset of size {train_array.shape} and generated dataset of size {gen_array.shape}.')
   
    
    #compute power spectra of the dataset and generated samples
    [k_x,  power_spectrum_avg_dataset]=zonal_averaged_power_spectrum_mod(train_array, time_avg=True)
    [k_x,  power_spectrum_avg_samples]=zonal_averaged_power_spectrum_mod(gen_array, time_avg=True)

    # plot both spectrum
    fig,ax = plt.subplots()
    ax.plot(k_x,  power_spectrum_avg_dataset[1:],label='dataset')
    ax.plot(k_x,  power_spectrum_avg_samples[1:],label='generated sample')
    ax.set_yscale('log')
    ax.set_xlabel('wavenumber',fontsize=18)
    ax.set_ylabel('power spectra',fontsize=18)
    ax.set_title(args.expid,fontsize=16)
    ax.legend(fontsize=15)
    fileo="powerspectra_"+str(args.num_samples)+"_"+args.expid+".png"
    plt.savefig(fileo)

    #compute variance and ACC in time and space
    mean_dataset=np.mean(train_array,axis=0)
    var_dataset=np.var(train_array.flatten())
    var_gen=np.var(gen_array.flatten())
    #prmse=np.sqrt(np.mean((train_array.flatten-gen_array.flatten())**2))
    #pprint(f'Variance of dataset: {var_dataset} and generated dataset  {var_gen}.')
    #pprint(f'rmse: {rmse} ')
    
   #compute variance and ACC in  space for each timestep
    print("varience of each timestep")
    tacc_res=[]

    for itime in range(train_array.shape[0]):
        var_dataset=np.var(train_array[itime,:,:].flatten())
        var_gen=np.var(gen_array[itime,:,:].flatten())
        print(f'timestep: {itime}, Variance of dataset: {var_dataset} and generated dataset  {var_gen}.')
        #compute ACC
        acc_res=ACC(gen_array[itime,:,:],train_array[itime,:,:],mean_dataset)
        print(f'timestep: {itime}, ACC: {acc_res} ')
        tacc_res.append(acc_res)
    tacc_res=np.array(tacc_res)    
    print(f'ACC , mean: {np.mean(tacc_res)}')
    with open('ACC_'+args.expid+'_.txt', 'w') as output:
        output.write(str(np.mean(tacc_res)))

  

    #temporal corr and rmse map
    #fileo="maprmse_"+str(args.num_samples)+".png"
    #map_rmse_cor(dataset,sample,fileo)

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_filepath", type=str, default="", help="Path to the training data file")
    parser.add_argument("--gen_filepath", type=str, default="", help="Path to the gen data file")
    parser.add_argument("--conditional", type=bool, default=False, help="Conditional Data?")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--expid", type=str, default="", help="")
    args = parser.parse_args()
    main(args) 
    
 
