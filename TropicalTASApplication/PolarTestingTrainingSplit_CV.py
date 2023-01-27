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
    ensemble_train_indices = random.sample(range(0,Nensembles),10)

    # Select natural and forced trends as well as the training and testing data
    trend_data = xarray_file.to_array()[0]
    #NatTrendsTrain = trend_data[ensemble_train_indices,0].to_numpy()
    #ForTrendsTrain = trend_data[ensemble_train_indices,1].to_numpy()
    NatTrendsTrain = trend_data[:,0].to_numpy()
    ForTrendsTrain = trend_data[:,1].to_numpy()

    return(NatTrendsTrain, ForTrendsTrain)

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

def tropical_mean_trend(trends, land_sea_mask, weights, latbounds):
    """ 
    Takes map of trends and finds average over the 30S-30N region.
    """
    # Find land and ocean trends using land-sea mask
    ocean_trends = np.array([np.ma.masked_array(data=trends[i], mask=land_sea_mask, fill_value=np.nan).filled() for i in range(len(trends))])
    land_trends = np.array([np.ma.masked_array(data=trends[i], mask=abs(land_sea_mask-1), fill_value=np.nan).filled() for i in range(len(trends))])
    
    # Also find land and ocean weights using same mask
    weights_map = np.reshape(np.tile(weights, (144)), (144,72)).T
    ocean_weights = np.ma.masked_array(data=weights_map, mask=land_sea_mask, fill_value=np.nan).filled()
    land_weights = np.ma.masked_array(data=weights_map, mask=abs(land_sea_mask-1), fill_value=np.nan).filled()
    length_lat_bounds = latbounds[1] - latbounds[0]
    
    # reshape trend maps
    ReshapedTrends = np.reshape(trends[:,latbounds[0]:latbounds[1],:], (np.shape(trends)[0],length_lat_bounds*144))
    ReshapedOceans = np.reshape(ocean_trends[:,latbounds[0]:latbounds[1],:], (np.shape(ocean_trends)[0],length_lat_bounds*144))
    ReshapedLands = np.reshape(land_trends[:,latbounds[0]:latbounds[1],:], (np.shape(land_trends)[0],length_lat_bounds*144))

    # Get average trends over total, ocean, and land
    TropicalAverageTrend = np.nansum(ReshapedTrends, axis=1)/(np.nansum(weights[latbounds[0]:latbounds[1]])*144)
    TropicalOceanTrend = np.nansum(ReshapedOceans, axis=1)/(np.nansum(ocean_weights[latbounds[0]:latbounds[1]]))
    TropicalLandTrend = np.nansum(ReshapedLands, axis=1)/(np.nansum(land_weights[latbounds[0]:latbounds[1]]))
    
    return(TropicalAverageTrend, TropicalOceanTrend, TropicalLandTrend)

def is_land(x, y):
    """
    Uses knowledge of land points to decide whether given location 
    is land or ocean. X is longitude and Y is latitude.
    """
    return land.contains(sgeom.Point(x, y))*1

def training_testing_split(path_to_data):
    # set path to data
    ModelDataFiles = glob.glob(path_to_data)

    # select lat bounds used for subsection of globe
    latbounds = [28,44]
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
    land_sea_mask = np.concatenate([land_sea_mask[:,72:], land_sea_mask[:,:72]], axis=1)
    
    TrainingPredictorData = []
    TrainingTargetData = []
    TestingPredictorData = []
    TestingTargetData = []
    TestingTotalTrend = []
    for x in range(len(ModelDataFiles)):
        TrainingModelDataFiles = ModelDataFiles[:x] + ModelDataFiles[x+1:]
        TestingModelDataFiles = ModelDataFiles[x] 

        # First, take care of training data
        #########################--------Training--------#########################
        OneCVTrainingPredictorData = []
        OneCVTrainingTargetData = []
        for datafile in TrainingModelDataFiles:
            xarray_file = xr.open_dataset(datafile)
            
            # find training data for natural and forced trends
            NatTrendsTrain, ForTrendsTrain = train_test_splitting(xarray_file)

            # reshape trends so that trend maps from different time periods and ensembles are treated equal
            NatTrendsTrain = model_ensemble_reshaper(NatTrendsTrain)
            ForTrendsTrain = model_ensemble_reshaper(ForTrendsTrain)

            # weight trend maps by cosine of latitude
            weights = np.cos(np.deg2rad(latitudes)) # these will be used to weight predictors
            NatTrendsTrain_weighted = np.multiply(NatTrendsTrain, weights[np.newaxis,:,np.newaxis])
            ForTrendsTrain_weighted = np.multiply(ForTrendsTrain, weights[np.newaxis,:,np.newaxis])
            
            # true trend maps are sum of natural and forced trends
            TrueTrendsTrain = NatTrendsTrain_weighted + ForTrendsTrain_weighted
            
            # reshape predictors as vector
            TrainingTrends_vectors = predictor_reshaper(TrueTrendsTrain)

            # find tropical mean trend value
            NatTrendsTrainTropicalMean = np.transpose(tropical_mean_trend(NatTrendsTrain_weighted, land_sea_mask, weights, latbounds))
            ForTrendsTrainTropicalMean = np.transpose(tropical_mean_trend(ForTrendsTrain_weighted, land_sea_mask, weights, latbounds))

            # append to training data 
            [OneCVTrainingPredictorData.append(TrainingTrends_vectors[i]) for i in range(len(TrainingTrends_vectors))]
            [OneCVTrainingTargetData.append([NatTrendsTrainTropicalMean[i], ForTrendsTrainTropicalMean[i]]) 
            for i in range(len(ForTrendsTrainTropicalMean))]


        # Now, curate the testing data
        #########################--------Testing--------#########################

        # we only need one model of testing for this CV iteration
        test_data = xr.open_dataset(TestingModelDataFiles).to_array()[0]
        
        # reshape trends so that trend maps from different time periods and ensembles are treated equal
        NatTrendsTest = test_data[:,0].to_numpy()
        ForTrendsTest = test_data[:,1].to_numpy()


        # reshape trends so that trend maps from different time periods and ensembles are treated equal
        NatTrendsTest = NatTrendsTest[:,-1]#model_ensemble_reshaper(NatTrendsTest)
        ForTrendsTest = ForTrendsTest[:,-1]#model_ensemble_reshaper(ForTrendsTest)

        # weight trend maps by cosine of latitude, you can use the weights from above
        NatTrendsTest_weighted = np.multiply(NatTrendsTest, weights[np.newaxis,:,np.newaxis])
        ForTrendsTest_weighted = np.multiply(ForTrendsTest, weights[np.newaxis,:,np.newaxis])

        # true trend maps are sum of natural and forced trends
        TrueTrendsTest = NatTrendsTest_weighted + ForTrendsTest_weighted
        
        # reshape predictors as vector
        OneCVTestingPredictorData = predictor_reshaper(TrueTrendsTest)
        OneCVTestingPredictorData_reshaped = np.reshape(OneCVTestingPredictorData, (np.shape(OneCVTestingPredictorData)[0],72,144))[:,latbounds[0]:latbounds[1],:]
        OneCVTestingPredictorData_reshaped = np.reshape(OneCVTestingPredictorData_reshaped, (np.shape(OneCVTestingPredictorData_reshaped)[0], (latbounds[1]-latbounds[0])*144))
        OneCVTestingPredictorDataAverageTrend = np.nansum(OneCVTestingPredictorData_reshaped, axis=1)/(np.nansum(weights[latbounds[0]:latbounds[1]])*144)

        # find tropical mean trend value
        NatTrendsTestTropicalMean = np.transpose(tropical_mean_trend(NatTrendsTest_weighted, land_sea_mask, weights, latbounds))
        ForTrendsTestTropicalMean = np.transpose(tropical_mean_trend(ForTrendsTest_weighted, land_sea_mask, weights, latbounds))

        # reshape the target variables 
        OneCVTestingTargetData= np.swapaxes([NatTrendsTestTropicalMean, ForTrendsTestTropicalMean], 0,1)

        TrainingPredictorData.append(OneCVTrainingPredictorData)
        TrainingTargetData.append(OneCVTrainingTargetData)
        TestingPredictorData.append(OneCVTestingPredictorData)
        TestingTargetData.append(OneCVTestingTargetData)
        TestingTotalTrend.append(OneCVTestingPredictorDataAverageTrend)

    return(TrainingPredictorData, TrainingTargetData, TestingPredictorData, TestingTargetData, TestingTotalTrend)
