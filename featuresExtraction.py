# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
import pandas as pd 
import numpy as np
from features.colorHistFeats import histFeats , calcDistanceHist , extractFeaturesHistDatabaseImg 
from features.colorDominantFeats import colorDominantFeats , calcDistanceColorDom , extractFeaturesColDomDatabaseImg  
from features.textureHaralickFeats import haralickTextureFeats , calcDistanceHaralickTexture , extractFeaturesHaralickDatabaseImg
from features.filterGaborFeats import filtreGaborFeats , calcDistanceGabor , extractFeaturesGaborDatabaseImg
from methodes import load_images_from_folder , getIndexImages , distanceTotal
from methodes import saveResults  , normalize , plotHist , plotColorDom
from scipy.spatial.distance import euclidean
from features.textureFourrierFeats import extractFeaturesFourierDescriptorDatabaseImg


#load data
dataImages = load_images_from_folder('static/dataset/beaver')

# extract histogram  features --> save to json file 'df_hist.json'
featuresHistDatabase = extractFeaturesHistDatabaseImg(dataImages)

# extract color dominant features --> save to json file 'df_colorDom.json'
nbreDominantColors = 1
featuresColDomDatabase = extractFeaturesColDomDatabaseImg( dataImages , nbreDominantColors)

# extract haralic texture features --> save to json file 'df_haralickTexture.json'
featuresHaralickDatabase = extractFeaturesHaralickDatabaseImg( dataImages)

