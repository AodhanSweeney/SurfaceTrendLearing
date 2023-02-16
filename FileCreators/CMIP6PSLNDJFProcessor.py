#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xarray as xr
import zipfile
import glob
import netCDF4 as nc
import xesmf as xe
from scipy import stats

def trend_finder(timeseries_maps):
    """
    Find decadal trends at each lat and lon point. 
    """
    shape_of_maps = np.shape(timeseries_maps)
    # must be adjusted based on len of time used for trends
    years = np.linspace(0,3.5,36)
    trend_map = []
    for lat in range(0, shape_of_maps[1]):
        lat_line = []
        for lon in range(0, shape_of_maps[2]):
            timeseries = timeseries_maps[:,lat,lon]
            trend = stats.linregress(x=years, y=timeseries)[0] # in K/dec
            lat_line.append(trend)
        trend_map.append(lat_line)
    return(np.array(trend_map))
            
# load path to CMIP 6 data
path_to_CMIP6_data = '/home/disk/pna2/aodhan/CMIP6/historical_monthly_psl_google/*'
model_paths = glob.glob(path_to_CMIP6_data)

# here we select the start times of trends, this can be changed
start_times = np.arange(1854,1980,5)

# for each model...
for model in model_paths:
    realizations = glob.glob(model + '/*')

    # create data dictionary
    model_trends = []
    
    # for each realization...
    simulation_index = []
    for ensemble_member in realizations:
        simulation = xr.open_dataset(ensemble_member)
        model_name = simulation.source_id
        simulation_name = simulation.variant_label
        simulation_number = int(simulation_name.split('i')[0][1:])
        simulation_index.append(simulation_number)
        print(model_name, simulation_name, simulation_number)

         # CMIP6 models must be regridded, below we define input and output grids
        latitudes = simulation.lat.values
        longitudes = simulation.lon.values
        InputGrid = {"lon": longitudes, "lat": latitudes}
        OutputGrid = {"lon": np.arange(1.25, 358.751, 2.5), "lat": np.arange(-88.75, 88.751, 2.5)}
        regridder = xe.Regridder(InputGrid, OutputGrid, "bilinear")
        
        # create lists for where to append data
        all_timeperiod_trends = []
        time_keys = []

        # find time slices for just winter
        for i in start_times:
            start_date = str(i) + '-01'
            end_date = str(i+35) + '-12'

            # find correct time period of data
            time_slice_data = simulation.sel(time=slice(start_date, end_date))
            data_array = time_slice_data.psl.values

            # find trends for each of the time periods
            shape_of_data_array = np.shape(data_array)
            data_calendar = np.reshape(data_array, (36, 12, shape_of_data_array[1], shape_of_data_array[2]))
            season_calendar = [data_calendar[:,10],data_calendar[:,11], data_calendar[:,0], 
                               data_calendar[:,1], data_calendar[:,2]]
            timeseries_map = np.nanmean(season_calendar, axis=0)
            trend_map = trend_finder(timeseries_map)
            trend_map_2p5x2p5 = regridder(trend_map)
            all_timeperiod_trends.append(trend_map_2p5x2p5)
            time_keys.append(i)
        
        # append all timeperiod trends 
        model_trends.append(all_timeperiod_trends)

    # Timeperiod data will be dumped into NetCDF files
    fileName = path_to_CMIP6_data[:-2] + '/trendmap/' + model_name.replace("-", "_") + '_PSL_NDJFM_TrendMaps.nc'

    # Create netcdf file with dimensions
    ds = nc.Dataset(fileName, 'w', format='NETCDF4')
    ensemble_member = ds.createDimension('ensemble_member', len(simulation_index))
    TrendTimePeriod = ds.createDimension('TrendTimePeriod', len(time_keys)) # 26 timeperiods
    Lat = ds.createDimension('Lat', 72)
    Lon = ds.createDimension('Lon', 144)

    # Add variables to dimensions
    ensemble_member = ds.createVariable('ensemble_member', int, ('ensemble_member',))
    TrendTimePeriod = ds.createVariable('TrendTimePeriod', int, ('TrendTimePeriod',))
    Lat = ds.createVariable('Lat', 'f4', ('Lat',))
    Lon = ds.createVariable('Lon', 'f4', ('Lon',))
    Ts_trends = ds.createVariable('ts_trend', 'f4', ('ensemble_member', 'TrendTimePeriod', 'Lat', 'Lon'))

    # Assing values to variables
    ensemble_member[:] = simulation_index
    TrendTimePeriod[:] = time_keys
    Lat[:] = np.arange(-88.75, 88.751, 2.5)
    Lon[:] = np.arange(1.25, 358.751, 2.5)
    Ts_trends[:] = model_trends

    ds.close()
