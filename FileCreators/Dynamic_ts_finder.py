import numpy as np
import glob
import xesmf as xe
import netCDF4 as nc
import os
from sklearn.cross_decomposition import PLSRegression

def polar_winter_temps(tas_data_regridded):
    polar_temps = tas_data_regridded.sel(lat=slice(60, 90)).tas.values
    weights = np.cos(np.deg2rad(tas_data_regridded.lat.sel(lat=slice(60,90)).values))
    weighted_polar_temps = np.multiply(polar_temps, weights[np.newaxis,:,np.newaxis])
    avg_polar_temps = np.nansum(np.reshape(weighted_polar_temps, (1980, 12*144)), axis=1)/(np.nansum(weights)*144)
    avg_polar_temps_cal = np.reshape(avg_polar_temps, (165,12))
    ndjfm_polar_cal = [avg_polar_temps_cal[:,0], avg_polar_temps_cal[:,1], avg_polar_temps_cal[:,2], 
                       avg_polar_temps_cal[:,10], avg_polar_temps_cal[:,11]]
    ndjfm_polar_cal = np.swapaxes(ndjfm_polar_cal, 0,1)
    return(ndjfm_polar_cal)

def nh_winter_press(psl_data_regridded):
    nh_psl = psl_data_regridded.sel(lat=slice(20, 90)).psl.values
    nh_psl_cal = np.reshape(nh_psl, (165,12,np.shape(nh_psl)[1], np.shape(nh_psl)[2]))
    ndjfm_nh_psl_cal = [nh_psl_cal[:,0], nh_psl_cal[:,1], nh_psl_cal[:,2], nh_psl_cal[:,10], nh_psl_cal[:,11]]
    ndjfm_nh_psl_cal = np.swapaxes(ndjfm_nh_psl_cal, 0,1)
    return(ndjfm_nh_psl_cal)

# find all potential models
psl_models = glob.glob('/home/disk/pna2/aodhan/CMIP6/historical_monthly_psl_google/*')

for model_path in psl_models[1:]:
    model_paths = model_path + '/*'
    psl_files = glob.glob(model_paths)

    psl_and_tas_files = []
    for psl_file in psl_files:
        tas_file = psl_file[:48] + 'tas' + psl_file[51:]
        psl_and_tas_files.append([psl_file, tas_file])

    # define times for final netcdf
    winter_times = np.arange(1850,2015,1)

    true_and_dynamic_ts = []
    for file_set in psl_and_tas_files:
        try:
            psl_data = xr.open_dataset(file_set[0])
            tas_data = xr.open_dataset(file_set[1])
        except:
            print('Error opening file set: ', file_set)
        print('Opened file set: ', file_set[0])
        # CMIP6 models must be regridded, below we define input and output grids
        latitudes = psl_data.lat.values # psl and tas have same grid
        longitudes = psl_data.lon.values
        InputGrid = {"lon": longitudes, "lat": latitudes}
        OutputGrid = {"lon": np.arange(1.25, 358.751, 2.5), "lat": np.arange(-88.75, 88.751, 2.5)}
        regridder = xe.Regridder(InputGrid, OutputGrid, "bilinear", periodic=True)
        psl_data_regridded = regridder(psl_data)
        tas_data_regridded = regridder(tas_data)

        # get polar temperature during winter
        ndjfm_polar_temps_cal = polar_winter_temps(tas_data_regridded)

        # get NH pressure data during winter then weight this by latitude
        ndjfm_nh_psl_cal = nh_winter_press(psl_data_regridded)
        
        # preform cross validation of dynamic adjustment
        dynamical_contributions = []
        for x in range(165):
            temp_minus_one_winter = np.delete(ndjfm_polar_temps_cal, x, axis=0)
            temp_minus_one_winter = np.nanmean(temp_minus_one_winter, axis=1)
            pres_minus_one_winter = np.delete(ndjfm_nh_psl_cal, x, axis=0)
            pres_minus_one_winter = np.nanmean(pres_minus_one_winter, axis=1)
            
            # scale X train data 
            pres_minus_one_winter_ts = np.reshape(pres_minus_one_winter, (164,28,144))
            pres_minus_one_winter_mean = np.nanmean(pres_minus_one_winter_ts, axis=0)
            pres_minus_one_winter_mr = pres_minus_one_winter_ts - pres_minus_one_winter_mean
            pres_minus_one_winter_std = np.nanstd(pres_minus_one_winter_mr, axis=0)
            pres_minus_one_winter_scaled = pres_minus_one_winter_mr/pres_minus_one_winter_std

            # scale X test data
            pres_all_ts = np.nanmean(ndjfm_nh_psl_cal, axis=1)
            pres_all_mr = pres_all_ts - pres_minus_one_winter_mean
            pres_all_scaled = pres_all_mr/pres_minus_one_winter_std

            # weight X data by latitude
            weights = np.cos(np.deg2rad(psl_data_regridded.lat.sel(lat=slice(20,90)).values))
            pres_minus_one_winter_weighted = np.multiply(pres_minus_one_winter_scaled, weights[np.newaxis, :,np.newaxis])
            pres_all_scaled_weighted = np.multiply(pres_all_scaled, weights[np.newaxis, :,np.newaxis])

            # scale Y data
            temp_minus_one_winter_ts = np.reshape(temp_minus_one_winter, (164))
            temp_minus_one_winter_mean = np.nanmean(temp_minus_one_winter_ts, axis=0)
            temp_minus_one_winter_mr = temp_minus_one_winter_ts - temp_minus_one_winter_mean
            temp_minus_one_winter_std = np.nanstd(temp_minus_one_winter_mr, axis=0)
            temp_minus_one_winter_scaled = temp_minus_one_winter_mr/temp_minus_one_winter_std

            # define X and Y data
            X = np.reshape(pres_minus_one_winter_weighted, (164, 28*144))
            Y = np.reshape(temp_minus_one_winter_scaled, (164))
            
            # create PLS model with 2 components
            pls = PLSRegression(n_components=2, scale='False')
            pls.fit(X, Y)

            # deploy on all pressure data
            all_pressures = np.reshape(pres_all_scaled_weighted, (165, 28*144))
            temp_dynamical = pls.predict(all_pressures)

            # unscale the data so that units are again in K
            temp_dynamical_multiplied_by_std = temp_dynamical*temp_minus_one_winter_std
            dynamical_contributions.append(temp_dynamical_multiplied_by_std)
            break

        # find mean dynamical contribution over all cross validations
        dynamical_mean = np.nanmean(dynamical_contributions, axis=0)[:,0]

        # reshape the raw polar timeseries
        polar_temp_timeseries = np.nanmean(ndjfm_polar_temps_cal, axis=1)

        # create a netcdf file
        storage_path = '/home/disk/pna2/aodhan/CMIP6/historical_dynamical_ts'
        completed_models = glob.glob(storage_path + '/*')
        completed_model_strings = [completed_models[i].split('/')[7] for i in range(0, len(completed_models))]
        model = file_set[0].split('/')[7]
        if model not in completed_model_strings:
            os.mkdir(storage_path + '/' + model)
        simulation = file_set[0].split('/')[8]
        fileName = storage_path + '/' + model + '/' + simulation + '.nc'
        ds = nc.Dataset(fileName, 'w', format='NETCDF4')
        
        DynamicalContribution = ds.createDimension('DynamicalContribution', 2)
        time = ds.createDimension('time', 165)

        # Add variables to dimensions
        DynamicalContribution = ds.createVariable('DynamicalContribution', int, ('DynamicalContribution',))
        time = ds.createVariable(varname='time', datatype=int, dimensions=('time',))
        timeseries = ds.createVariable('temp', 'f4', ('DynamicalContribution', 'time'))

        # Assing values to variables
        DynamicalContribution[:] = [0,1]
        time[:] = winter_times
        timeseries[:] = [polar_temp_timeseries, dynamical_mean]

        # close netcdf    
        ds.close()
