import numpy as np
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
from scipy import stats

def find_trends(polar_ts, dynamic_ts):
    # remove dynamic contribution
    dynamically_adjusted = polar_ts - dynamic_ts
    
    # find decadal trends
    total_trend = stats.linregress(x=np.arange(0,3.6,0.1), y=polar_ts)[0]
    dynamically_adjusted_trend = stats.linregress(x=np.arange(0,3.6,0.1), y=dynamically_adjusted)[0]
    dynamic_contribution_trend = stats.linregress(x=np.arange(0,3.6,0.1), y=dynamic_ts)[0]

    return(total_trend, dynamically_adjusted_trend, dynamic_contribution_trend)

def training_testing():
    # find paths to data
    tas_data = np.sort(glob.glob('/home/disk/pna2/aodhan/CMIP6/historical_monthly_tas_google/trendmap/*'))
    psl_data = np.sort(glob.glob('/home/disk/pna2/aodhan/CMIP6/historical_monthly_psl_google/trendmap/*'))
    dynamical_data = np.sort(glob.glob('/home/disk/pna2/aodhan/CMIP6/historical_dynamical_ts/*'))

    # get name of each of the models to use
    models = [data_path.split('/')[-1].split('_TAS')[0] for data_path in tas_data]

    all_models_trends = []
    all_models_trend_maps = []
    for model_name in models:
        print(model_name)
        # find trend map files
        model_tas = tas_data[[model_name in tas_path for tas_path in tas_data]][0]
        model_psl = psl_data[[model_name in psl_path for psl_path in psl_data]][0]

        # load trend maps
        model_tas_trendmaps = xr.open_dataset(model_tas)
        model_psl_trendmaps = xr.open_dataset(model_psl)
        
        # find files for dynamic timeseries
        model_dyn_ts = dynamical_data[[model_name.replace('_', '-') in dyn_path for dyn_path in dynamical_data]][0]
        model_dyn_ts = glob.glob(model_dyn_ts + '/*_LandPoleward40.nc')
        dyn_simulation_indices = [int(simulation_path.split('/')[-1].split('i')[0][1:]) for simulation_path in model_dyn_ts]

        
        all_simulation_trends = []
        all_simulation_trend_maps = []
        for simulation_idx in model_tas_trendmaps.ensemble_member.values:
            try:
                # find specific simulation number for trend maps
                tas_simulation_trends = model_tas_trendmaps.sel(ensemble_member = simulation_idx).ts_trend.values
                psl_simulation_trends = model_psl_trendmaps.sel(ensemble_member = simulation_idx).ts_trend.values
                simulation_trend_maps = np.array([tas_simulation_trends, psl_simulation_trends])

                # find simulation for dynamic ts
                dyn_simulation_idx = np.where(simulation_idx == dyn_simulation_indices)[0][0]
                dyn_simulation_file = model_dyn_ts[dyn_simulation_idx]
                dyn_simulation_ts = xr.open_dataset(dyn_simulation_file)
            except:
                print('Could not find all files for simulation #', simulation_idx)
                continue
            
            # find trends associated with dynamic ts and
            simulation_trends = []
            start_times = np.arange(1854,1980,5)
            for year in start_times:
                start_date = year
                end_date = year+35
                time_slice_data = dyn_simulation_ts.sel(time=slice(start_date, end_date))
                polar_ts = time_slice_data.temp.values[0]
                dynamic_ts = time_slice_data.temp.values[1]
                total_trend, dynamically_adjusted_trend, dynamic_contribution_trend = find_trends(polar_ts, dynamic_ts)
                trends = [total_trend, dynamically_adjusted_trend, dynamic_contribution_trend]
                simulation_trends.append(trends)
            all_simulation_trends.append(simulation_trends)
            all_simulation_trend_maps.append(simulation_trend_maps)
        
        # forced components of trends are mean over all ensemble members
        forced_components = np.nanmean(all_simulation_trends, axis=0)

        # natural components of trends are difference between each ensemble member and forced components
        natural_components = np.array([ensemble_trends - forced_components for ensemble_trends in all_simulation_trends])
        
        # natural and forced components
        nat_for_trends = [[natural_components[i], forced_components] for i in range(0, len(natural_components))] 
        
        # append data 
        all_models_trends.append(nat_for_trends)
        all_models_trend_maps.append(all_simulation_trend_maps)
    return(all_models_trends, all_models_trend_maps)