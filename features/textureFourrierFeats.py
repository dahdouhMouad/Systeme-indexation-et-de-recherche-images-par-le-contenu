
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
import mahotas as mt
from pandas import DataFrame 
from scipy.spatial.distance import euclidean
from methodes import normalize

import numpy

MIN_DESCRIPTOR = 18  # surprisingly enough, 2 descriptors are already enough
TRAINING_SIZE = 100


def findDescriptor(img):
    """ findDescriptor(img) finds and returns the
    Fourier-Descriptor of the image contour"""
    retval, sample1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    contour = []
    contour, hierarchy = cv2.findContours(
        sample1,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
        contour)
    contour_array = contour[0][:, 0, :]
    contour_complex = numpy.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = numpy.fft.fft(contour_complex)
    return fourier_result
    
# extract texture features from databse images
def extractFeaturesFourierDescriptorDatabaseImg(dataImages):
    feats = []
    print('features extracting ...')
    for img in dataImages:
        features = findDescriptor(img) 
        feats.append(features)
    df_haralick = DataFrame(feats)
    df_haralick.to_json(r'features/json/Fourrier.json' , orient='split')
    print('Fourrier texture features extracted -> path = features/json/df_Fourrier.json ')
    return df_haralick   

# calculate euclidian distance for Fourrier feats 
def calcDistanceFourrierTexture(queryImage , df_Fourrier) :
    featsVectors = df_Fourrier.values.tolist()
    distances = {}
    print('--------------- calculate distances for Fourrier Feats  -------------------')
    for i in range(len(featsVectors)):
        queryFeatures = findDescriptor(queryImage)
        imgFeatures = featsVectors[i]
        #imgFeatures = cv2.normalize(imgFeatures, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        dist = euclidean(queryFeatures,imgFeatures)
        print('dist (query image , image %d )'%(i+1)+'-----> %f'%dist)
        distances[i] = dist
    print('------------calculate distances for Fourrier Feats completed ---------')
    distances = normalize(distances , 25) 
    return distances
 