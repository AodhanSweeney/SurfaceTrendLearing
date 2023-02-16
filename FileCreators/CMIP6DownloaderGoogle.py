#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import gcsfs

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_ta = df.query("activity_id=='CMIP' & table_id == 'Amon' & variable_id == 'psl' & experiment_id == 'historical'")

# this only needs to be created once
gcs = gcsfs.GCSFileSystem(token='anon')

def ensemble_splitter(row):
  member_id = row.member_id
  member_id_split = member_id.split('i')
  return(member_id_split[0], member_id_split[1])

df_ta[['realization', 'other']] = df_ta.apply(ensemble_splitter, axis=1, result_type='expand')
df_ta_new_justrealizations = df_ta[df_ta['other'] == '1p1f1']
print(len(df_ta_new_justrealizations))

import os

path_to_store_data = '/home/disk/pna2/aodhan/CMIP6/historical_monthly_psl_google'
for source_id in df_ta_new_justrealizations.source_id.unique():
  one_model_ds = df_ta_new_justrealizations[df_ta_new_justrealizations['source_id'] == source_id]
  if len(one_model_ds) >= 10:    
    for i in range(0, len(one_model_ds)):
      zstore = one_model_ds.zstore.values[i]

      # create a mutable-mapping-style interface to the store
      mapper = gcs.get_mapper(zstore)

      # open it using xarray and zarr
      ds = xr.open_zarr(mapper, consolidated=True)
      model_name = one_model_ds.iloc[i].source_id
      model_directory = os.path.join(path_to_store_data, model_name)
      try:
        os.mkdir(model_directory)
        variant_id = one_model_ds.iloc[i].member_id
        ds.to_netcdf(model_directory + '/' + variant_id)
        print(model_name + ' ' + variant_id)
      except:
        variant_id = one_model_ds.iloc[i].member_id
        ds.to_netcdf(model_directory + '/' + variant_id)
        print(model_name + ' ' + variant_id)
        
  
      