# import statements
import numpy as np
import xarray as xr
import glob
import random
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep

# Set up cartopy shape file so that it can distinguish between land and ocean
land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)

def train_test_splitting(xarray_file):
    """
    Split model data into train and testing. 10 ensemble 
    members for each model are used for training as to 
    not over weight a given model. All remaining ensemble 
    members are used for testing.
    """
    # Find number of ensembles and random indicies of testing versus training ensembels
    Nensembles = len(xarray_file.ensemble_member)
    ensemble_train_indices = random.sample(range(0,Nensembles),9)
    ensemble_test_indices = list(set(list(range(0,Nensembles))).difference(ensemble_train_indices))

    # Select natural and forced trends as well as the training and testing data
    trend_data = xarray_file.to_array()[0]
    NatTrendsTrain = trend_data[ensemble_train_indices,0].to_numpy()
    NatTrendsTest = trend_data[ensemble_test_indices,0].to_numpy()
    ForTrendsTrain = trend_data[ensemble_train_indices,1].to_numpy()
    ForTrendsTest = trend_data[ensemble_test_indices,1].to_numpy()

    return(NatTrendsTrain, NatTrendsTest, ForTrendsTrain, ForTrendsTest)

def model_ensemble_reshaper(trends):
    """
    Takes a given models testing or trainging data and 
    reshapes it so that timeperiods from different ensembles
    of a given model are treated equally.
    """
    reshaped_trends = np.reshape(trends, (np.shape(trends)[0]*np.shape(trends)[1], 72,144))
    return(reshaped_trends)

def predictor_reshaper(trends):
    """
    Takes maps of trends and reshapes grid points into a vector.
    """
    PredictorVector = np.reshape(trends, (np.shape(trends)[0], np.shape(trends)[1]*np.shape(trends)[2]))
    return(PredictorVector)

def tropical_mean_trend(trends, land_sea_mask):
    """ 
    Takes map of trends and finds average over the 30S-30N region.
    """
    ocean_trends = np.array([np.ma.masked_array(data=trends[i], mask=land_sea_mask, fill_value=np.nan).filled() for i in range(len(trends))])
    land_trends = np.array([np.ma.masked_array(data=trends[i], mask=abs(land_sea_mask-1), fill_value=np.nan).filled() for i in range(len(trends))])

    ReshapedTrends = np.reshape(trends[:,24:48,:], (np.shape(trends)[0],24*144))
    TropicalAverageTrend = np.average(ReshapedTrends, axis=1)
    TropicalOceanTrend = np.nanmean(np.reshape(ocean_trends[:,24:48,:], (np.shape(ocean_trends)[0],24*144)), axis=1)
    TropicalLandTrend = np.nanmean(np.reshape(land_trends[:,24:48,:], (np.shape(land_trends)[0],24*144)), axis=1)

    return(TropicalAverageTrend, TropicalOceanTrend, TropicalLandTrend)

def is_land(x, y):
    """
    Uses knowledge of land points to decide whether given location 
    is land or ocean. X is longitude and Y is latitude.
    """
    return land.contains(sgeom.Point(x, y))*1
def training_testing_split():
    # set path to data
    path_to_data = '/home/disk/pna2/aodhan/SurfaceTrendLearning/*.nc'
    ModelDataFiles = glob.glob(path_to_data)

    # create land sea mask
    sample_grid = xr.open_dataset(ModelDataFiles[0]) 
    latitudes = sample_grid.Lat.to_numpy()
    longitudes = sample_grid.Lon.to_numpy() - 180
    land_sea_mask = []
    for x in longitudes:
        land_sea_mask_at_latitude = []
        for y in latitudes:
            land_sea_mask_at_latitude.append(is_land(x, y))
        land_sea_mask.append(land_sea_mask_at_latitude)
    land_sea_mask = np.transpose(land_sea_mask)
    TraingPredictorData = []
    TrainingTargetData = []
    TestingPredictorData = []
    TestingTargetData = []
    for datafile in ModelDataFiles:
        xarray_file = xr.open_dataset(datafile) 

        # find training and testing data for natural and forced trends
        NatTrendsTrain, NatTrendsTest, ForTrendsTrain, ForTrendsTest = train_test_splitting(xarray_file)

        # reshape trends so that trend maps from different time periods and ensembles are treated equal
        NatTrendsTrain = model_ensemble_reshaper(NatTrendsTrain)
        NatTrendsTest = model_ensemble_reshaper(NatTrendsTest)
        ForTrendsTrain = model_ensemble_reshaper(ForTrendsTrain)
        ForTrendsTest = model_ensemble_reshaper(ForTrendsTest)

        # weight trend maps by cosine of latitude
        weights = np.cos(np.deg2rad(latitudes)) # these will be used to weight predictors
        NatTrendsTrain_weighted = np.multiply(NatTrendsTrain, weights[np.newaxis,:,np.newaxis])
        NatTrendsTest_weighted = np.multiply(NatTrendsTest, weights[np.newaxis,:,np.newaxis])
        ForTrendsTrain_weighted = np.multiply(ForTrendsTrain, weights[np.newaxis,:,np.newaxis])
        ForTrendsTest_weighted = np.multiply(ForTrendsTest, weights[np.newaxis,:,np.newaxis])

        
        # true trend maps are sum of natural and forced trends
        TrueTrendsTrain = NatTrendsTrain_weighted + ForTrendsTrain_weighted
        TrueTrendsTest = NatTrendsTest_weighted + ForTrendsTest_weighted
        
        # reshape predictors as vector
        TrainingTrends_vectors = predictor_reshaper(TrueTrendsTrain)
        TestingTrends_vectors = predictor_reshaper(TrueTrendsTest)

        # find tropical mean trend value
        NatTrendsTrainTropicalMean = np.transpose(tropical_mean_trend(NatTrendsTrain_weighted, land_sea_mask))
        NatTrendsTestTropicalMean = np.transpose(tropical_mean_trend(NatTrendsTest_weighted, land_sea_mask))
        ForTrendsTrainTropicalMean = np.transpose(tropical_mean_trend(ForTrendsTrain_weighted, land_sea_mask))
        ForTrendsTestTropicalMean = np.transpose(tropical_mean_trend(ForTrendsTest_weighted, land_sea_mask))

        [TraingPredictorData.append(TrainingTrends_vectors[i]) for i in range(len(TrainingTrends_vectors))]
        [TrainingTargetData.append([NatTrendsTrainTropicalMean[i], ForTrendsTrainTropicalMean[i]]) 
        for i in range(len(ForTrendsTrainTropicalMean))]
        [TestingPredictorData.append(TestingTrends_vectors[i]) for i in range(len(TestingTrends_vectors))]
        [TestingTargetData.append([NatTrendsTestTropicalMean[i], ForTrendsTestTropicalMean[i]]) 
        for i in range(len(ForTrendsTestTropicalMean))]

    TraingPredictorData = np.array(TraingPredictorData)
    TrainingTargetData = np.reshape(TrainingTargetData, (np.shape(TrainingTargetData)[0], 
                                    np.shape(TrainingTargetData)[1]*np.shape(TrainingTargetData)[2]))
    TestingPredictorData = np.array(TestingPredictorData)
    TestingTargetData = np.reshape(TestingTargetData, (np.shape(TestingTargetData)[0], 
                                    np.shape(TestingTargetData)[1]*np.shape(TestingTargetData)[2]))
    return(TraingPredictorData, TrainingTargetData, TestingPredictorData, TestingTargetData)
