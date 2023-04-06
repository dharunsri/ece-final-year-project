import os

from google.cloud import vision
from google.cloud.vision_v1 import types

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'deft-orb-382017-e67247b740b7.json'

client = vision.ImageAnnotatorClient()

def predict(file):
    content=file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    text = texts[0].description 
    
    return text
