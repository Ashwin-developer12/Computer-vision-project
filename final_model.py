from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pytesseract
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load and preprocess images
image_paths = ['Library-4.jpg', 'test_img.jpg',"ans.jpg","Hotpot.png","peter.png"]
images = [cv.imread(path) for path in image_paths]
preprocessed_images = [tf.keras.applications.efficientnet.preprocess_input(np.array(img)) for img in images]


model = VisionEncoderDecoderModel.from_pretrained('vit-gpt2-image-captioning')
feature_extractor = ViTImageProcessor.from_pretrained(
    'vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('vit-gpt2-image-captioning')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {'max_length': max_length, 'num_beams': num_beams}
o=1

def predict_step(image):
    pixel_values = feature_extractor(
        images=[image], return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]

image_captions =[]
for img in images:
    image_captions.append(predict_step(image=img))


scene_classification_model = tf.keras.applications.ResNet50(weights='imagenet')
image_categories = []
for img in preprocessed_images:
    img_resized = tf.image.resize(img, (224, 224))
    prediction = scene_classification_model.predict(np.expand_dims(img_resized, axis=0))
    predicted_class = tf.keras.applications.resnet.decode_predictions(prediction)[0][0][1]
    image_categories.append(predicted_class)
    


extracted_texts = [pytesseract.image_to_string(img) for img in images]


model = YOLO("yolov8n.pt")
people=[]
for img in images:
    det = model.predict(source = img)
    dat = det[0].boxes.boxes
    try:
        dat_box = pd.DataFrame(dat).astype(float)
   
        # for index, rows in dat_box.iterrows():
        #     print(index)  
        #     print(rows)
            # x = 
            # cv.rectangle(img, ())


        pr = pd.DataFrame(dat_box[(dat_box[5]==0) & (dat_box[4]>=0.45)])

        # print(dat_box[dat_box[4]>=0.45])
        count = pr.shape[0]
        people.append(count)
    except:
        count =0
        people.append(0)
        total = "no of persons: 0"
        cv.putText(img, total , (30, 30), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),1 )
       
        
    try:
        
        for index, rows in pr.iterrows():
                x = int(rows[0])
                y =int(rows[1])
                w = int(rows[2])
                h = int(rows[3])
                c = "person"
                cv.rectangle(img, (x,y), (w,h), (0, 0, 255), 3)
                cv.putText(img, str(c), (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                
        total = "no of persons:"+str(count)
        cv.putText(img, total , (30, 30), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),1 )
            
        
    except:
        total = "no of persons:"+str(count)
        cv.putText(img, total , (30, 30), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),1 )
            
          
        
               
        
        
        



for i in range(len(image_paths)):
    print(f"Image: {image_paths[i]}")
    src = image_paths[i]
    im = Image.open(src)
    im.show()
    print(f"Extracted Text: {extracted_texts[i]}")
    print(f"Number of People: {people[i]}")
    print(f"Image Category: {image_categories[i]}")
    print(f"Image Caption: {image_captions[i]}")
    print("-" * 30)