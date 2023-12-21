# Generic Libraries
from PIL import Image
import os
import pandas as pd
import numpy as np
import re,string,unicodedata

#Tesseract Library
import pytesseract

#Warnings
import warnings
warnings.filterwarnings("ignore")

#Garbage Collection
import gc

#Gensim Library for Text Processing
import gensim.parsing.preprocessing as gsp
from gensim import utils

#TextBlob Library (Sentiment Analysis)
from textblob import TextBlob, Word

#Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

#WordCloud Generator
from wordcloud import WordCloud,STOPWORDS
#Define Directory Path
sample_images = "C:/Users/steph/OneDrive/桌面/code/ml/dataTest/Sample Data Files"
test_images = "C:/Users/steph/OneDrive/桌面/code/ml/dataTest/Dataset"

def traverse(directory):
    path, dirs, files = next(os.walk(directory))
    fol_nm = os.path.split(os.path.dirname(path))[-1]
    print(f'Number of files found in "{fol_nm}" : ',len(files))
#Traversing the folders
traverse(sample_images)
traverse(test_images)
ex_txt = []   #list to store the extracted text

#Function to Extract Text
def TxtExtract(directory):
    """
    This function will handle the core OCR processing of images.
    """
    
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            print("testFilePath"+filepath)
            text = pytesseract.image_to_string(Image.open(filepath), timeout=5)
            if not text:
                ex_txt.extend([[file, "blank"]])
            else:   
                ex_txt.extend([[file, text]])
                
    fol_nm = os.path.split(os.path.dirname(subdir))[-1]
    
    print(f"Text Extracted from the files in '{fol_nm}' folder & saved to list..")
    #Extracting Text from JPG files in Sample Image Folder
TxtExtract(sample_images)

#Extracting Text from JPG files in Dataset Folder
TxtExtract(test_images)
#Converting the list to dataframe for further analysis
ext_df = pd.DataFrame(ex_txt,columns=['FileName','Text'])
#Inspect the dataframe
# ext_df.head()
print(ext_df)