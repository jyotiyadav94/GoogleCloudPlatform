import os, glob
import cv2
import os
import PIL
import re
import os
import pytesseract
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image as im
from scipy import ndimage
from difflib import SequenceMatcher
from itertools import groupby
from datasets import load_metric
from scipy.ndimage import interpolation as inter
from datasets import load_dataset
from datasets.features import ClassLabel
from transformers import AutoProcessor
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForTokenClassification
from transformers.data.data_collator import default_data_collator
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import LayoutLMv3ForTokenClassification,LayoutLMv3FeatureExtractor,LayoutLMv3ImageProcessor
import io
from typing import Dict, List, Any
from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor
import torch
from torchvision.transforms import functional as F
from transformers import AutoTokenizer, pipeline
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
from PIL import Image
import base64
import requests
import pytesseract
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


id2label={0: 'others', 1: 'fecha', 2: 'num_orden', 3: 'num_parte', 4: 'revision_ruta'}
class LayoutLMHandler(BaseHandler):
    def __init__(self):
        super(LayoutLMHandler, self).__init__()
        self.context = None
        model_dir="DataIntelligenceTeam/REPROCESO5.0"
        self.processor = AutoProcessor.from_pretrained(model_dir,apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.initialized = False
        

    def initialize(self, context):
        print('Model initialized')
        #model_dir = context.system_properties.get("model_dir")
        model_dir="DataIntelligenceTeam/REPROCESO5.0"
        self.processor = AutoProcessor.from_pretrained(model_dir,apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.initialized=True


    def process_image_pytesseract(self,image,width,height):
        width, height = image.size
        feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=True)
        encoding_feature_extractor = feature_extractor(image, return_tensors="pt",truncation=True)
        words, boxes = encoding_feature_extractor.words, encoding_feature_extractor.boxes
        return words,boxes


    # helper function to unnormalize bboxes for drawing onto the image
    def unnormalize_box(self,bbox, width, height):
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]



    def process_image_encoding(self,model, processor, image, words, boxes,width,height):
        # encode
        inference_image = [image.convert("RGB")]
        encoding = processor(inference_image ,words,boxes=boxes, truncation=True, return_offsets_mapping=True, return_tensors="pt", 
                        padding="max_length", stride =128, max_length=512, return_overflowing_tokens=True)
        offset_mapping = encoding.pop('offset_mapping')
        overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

        # change the shape of pixel values
        x = []
        for i in range(0, len(encoding['pixel_values'])):
          x.append(encoding['pixel_values'][i])
        x = torch.stack(x)
        encoding['pixel_values'] = x

        # forward pass
        outputs = model(**encoding)

        # get predictions
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoding.bbox.squeeze().tolist()

        # only keep non-subword predictions
        preds = []
        l_words = []
        bboxes = []
        token_section_num = [] 

        if (len(token_boxes) == 512):
          predictions = [predictions]
          token_boxes = [token_boxes]

        for i in range(0, len(token_boxes)):
          for j in range(0, len(token_boxes[i])):
            unnormal_box = self.unnormalize_box(token_boxes[i][j], width, height)
            if (np.asarray(token_boxes[i][j]).shape != (4,)):
              continue
            elif (token_boxes[i][j] == [0, 0, 0, 0] or token_boxes[i][j] == 0):
              #print('zero found!')
              continue
            # if bbox is available in the list, just we need to update text
            elif (unnormal_box not in bboxes): 
              preds.append(predictions[i][j])
              l_words.append(processor.tokenizer.decode(encoding["input_ids"][i][j]))
              bboxes.append(unnormal_box)
              token_section_num.append(i)
            else:
              # we have to update the word
              _index = bboxes.index(unnormal_box)
              if (token_section_num[_index] == i): 
                # check if they're in a same section or not (documents with more than 512 tokens will divide to seperate
                # parts, so it's possible to have a word in both of the pages and we have to control that repetetive words
                # HERE: because they're in a same section, so we can merge them safely
                l_words[_index] = l_words[_index] + processor.tokenizer.decode(encoding["input_ids"][i][j])
              else:
                continue  
        
        return bboxes, preds, l_words, image

    def process_form(self,json_df):

      labels = [x['LABEL'] for x in json_df]
      texts = [x['TEXT'] for x in json_df]
      cmb_list = []
      for i, j in enumerate(labels):
        cmb_list.append([labels[i], texts[i]])

      grouper = lambda l: [[k] + sum((v[1::] for v in vs), []) for k, vs in groupby(l, lambda x: x[0])]

      list_final = grouper(cmb_list)
      lst_final = []
      for x in list_final:
        json_dict = {}
        json_dict[x[0]] = (' ').join(x[1:])
        lst_final.append(json_dict)

      return lst_final

    
    def iob_to_label(self,label):
        return id2label.get(label, 'others')


    def visualize_image(self,final_bbox, final_preds, l_words, image):

          draw = ImageDraw.Draw(image)
          font = ImageFont.load_default()
          
          label2color = {'fecha':'red', 'num_orden':'yellow', 'num_parte':'lime','revision_ruta':'black', 'others':'violet'}
          f_labels ={'fecha':'red', 'num_orden':'yellow', 'num_parte':'lime','revision_ruta':'black', 'others':'violet'}

          json_df = []

          for ix, (prediction, box) in enumerate(zip(final_preds, final_bbox)):
            if prediction is not None:
              predicted_label = self.iob_to_label(prediction).lower()
            if predicted_label not in ["others"]:
              draw.rectangle(box, outline=label2color[predicted_label])
              draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)
            json_dict = {}
            json_dict['TEXT'] = l_words[ix]
            json_dict['LABEL'] = f_labels[predicted_label]
            json_df.append(json_dict)
          return image, json_df


    def add_dataframe(self,df_main):
      col_name_map = {'red': 'fecha', 'yellow': 'num_orden', 'lime': 'num_parte', 'black': 'revision_ruta'}
      columns = list(col_name_map.values())
      data = {col:[] for col in columns}
      for i in df_main:
          for k, v in i.items():
              if k in col_name_map:
                  data[col_name_map[k]].append(v)
                  
      # join the list of strings for each column and convert to a dataframe
      for col in columns:
          data[col] = [' '.join(data[col])]
      df_upper = pd.DataFrame(data)
      key_value_pairs = []
      for col in df_upper.columns:
          key_value_pairs.append({'key': col, 'value': df_upper[col][0]})
      df_key_value = pd.DataFrame(key_value_pairs)
      return df_key_value


    def preprocess(self, request):
        print('starting preprocess')
        # Extracting image bytes from the request
        image_bytes = base64.b64decode(request[0]["body"])
        
        # Opening the image from bytes
        image = Image.open(BytesIO(image_bytes))
        
        width,height=image.size

        words, boxes = self.process_image_pytesseract(image, width, height)

        bbox, preds, words, image = self.process_image_encoding(self.model, self.processor, image, words, boxes,width,height)
        im, df = self.visualize_image(bbox, preds, words, image)
        df_main = self.process_form(df) 
        print('ending preprocess')
        return df_main



    def inference(self, model_input):
        print('inference starting')
        df = self.add_dataframe(model_input)
        print('ending inference')

        return df


    def postprocess(self, inference_output):
        df=inference_output
        print('starting postprocess')
        #fecha_value = df.loc[df['key'] == 'fecha', 'value'].str.extract(r'(\d{2}\.\d{2}\.\d{4})', expand=False)
        print('completed postprocess')

        return df


_service = LayoutLMHandler()

def handle(data, context):
    data="/content/drive/MyDrive/torchmodel/sample.jpg"
    image_path = data
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Convert image bytes to base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Prepare request payload
    request_payload = [{"body": image_base64}]
    data_preproces = _service.preprocess(request_payload)
    data_inference = _service.inference(data_preproces)
    data_postprocess = _service.postprocess(data_inference)

    return data_postprocess
