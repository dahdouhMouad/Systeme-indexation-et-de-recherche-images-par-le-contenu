# -*- coding: utf-8 -*-
from flask import Flask, render_template, redirect, url_for, request
import cv2
import pandas as pd 
import os
from features.colorHistFeats import calcDistanceHist
from features.colorDominantFeats import calcDistanceColorDom 
from features.textureHaralickFeats import calcDistanceHaralickTexture 
from methodes import distanceTotal , getIndexImages , saveResults , filenames , saveToJson , clearResultsFolder
from random import seed
from random import randint
clearResultsFolder('./static/query_img/')
clearResultsFolder('./static/results/')
clearResultsFolder('distances')

#STATIC_FOLDER = './static'
app = Flask(__name__)


UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
  

@app.route('/')
def home():
    clearResultsFolder('./static/query_img/')
    clearResultsFolder('./static/results/')
    return render_template('index.html')

# features loading

similar_images=[]


@app.route("/output", methods=["GET", "POST"])
def upload_image():
    clearResultsFolder('./static/query_img/')
    clearResultsFolder('./static/results/')
    if request.method == 'POST':
        if request.files:
            #result=request.form.getlist("check")
            result=request.form["check"]
            print(result)
            
            # features loading
            featuresHistDatabase = pd.read_json('features/json/'+result+'/df_hist.json' , orient='split')
            featuresColDomDatabase= pd.read_json('features/json/'+result+'/df_colorDom.json' , orient='split')
            featuresHaralickDatabase= pd.read_json('features/json/'+result+'/df_haralick.json' , orient='split')
            #featuresGaborDatabase= pd.read_json('features/json/'+result+'/df_gabor.json' , orient='split')
            # features loading                        
            
            print("*************************************************************")
            
            #Reading the uploaded image
            
            image = request.files["image"]
            ue = str(randint(0, 1000000))
            imageFile = image.filename 
            ImgFile = ue+"Q"+imageFile
            filename = os.path.join('./static/query_img/',ImgFile )
            image.save(filename)
            queryImage = cv2.imread(filename)
            
            # Hist distance
            distanceHist = calcDistanceHist(queryImage , featuresHistDatabase)

            # ColorDom distance
            nbreDominantColors = 1
            distanceColDom = calcDistanceColorDom(queryImage  , nbreDominantColors  , featuresColDomDatabase)
            
            # haralick texture distance
            distanceHaralickTexture = calcDistanceHaralickTexture(queryImage , featuresHaralickDatabase)
            
            # Gabor filtre distance
            #distanceGabor = calcDistanceGabor(queryImage , featuresGaborDatabase)
            # distance global
            distanceGlobal = distanceTotal( distanceHist , distanceColDom , distanceHaralickTexture )
            # save distances to json file
            saveToJson(imageFile , distanceGlobal )
            
            # get index retrieved images
            indexImagesResults = getIndexImages(distanceGlobal , False)
            path_results = './static/results/'
            numberImagesToRetrieve = 10
            path_dataImages = './static/dataset/'+result
            # save retrieved images to folder results
            saveResults(path_results , numberImagesToRetrieve , indexImagesResults , path_dataImages)
            # load retrieved images
            
            similar_images = filenames(path_results)
            for i in similar_images:
                print(i)
            print(filename)
            sim3 =similar_images[0:3]
            simt=similar_images[4:9]
            
            
            return render_template("output.html", query_image = filename , similar_images3 = sim3 ,similar_imagest = simt)
            print('Done !') 
            
        else :
            return redirect(url_for('home'))
        



if __name__ == '__main__':
   app.run()
        
            