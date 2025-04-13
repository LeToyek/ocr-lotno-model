import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import easyocr
from io import StringIO
import numpy as np
from PIL import Image
from ultralytics import YOLO
import argparse
import re

reader = easyocr.Reader(["en"])


def process_cap(path:str):
  img = cv2.imread(path)
  model_cap_lot_number_full = YOLO("../runs/detect/cap_lot_number_full/weights/best.pt")
  image_cap_lot_number_full = img

  fresh_image = image_cap_lot_number_full.copy()
  results = model_cap_lot_number_full(image_cap_lot_number_full)
  bbox_data = []
  cropped_image = []
  for r in results:
      boxes = r.boxes
      leftest = 0
      for box in boxes:
          b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
          c = box.cls  # get class index
          confidences = box.conf  # get confidence
          bbox_data.append((b, model_cap_lot_number_full.names[int(c)], confidences))
  for bbox in bbox_data:
      cropped_image = fresh_image[
          int(bbox[0][1]) : int(bbox[0][3]), int(bbox[0][0]) : int(bbox[0][2])
      ]
  gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
  kernel_size = 3
  max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
  alpha = 0.8  # Contrast control (1.0-3.0)
  beta = 0  # Brightness control (0-100)

  bilateral_filtered_image = cv2.bilateralFilter(gray_image, 5, 175, 175)
  more_denoised = cv2.fastNlMeansDenoising(bilateral_filtered_image, None, 13, 7, 21)
  contrast = cv2.convertScaleAbs(more_denoised, alpha=alpha, beta=beta)
  thresh = cv2.adaptiveThreshold(
      contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
  )
  model_cap_more_train = YOLO(
      "../runs/detect/cap_more_train [09-11-2024]/weights/best.pt"
  )
  three_channel_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
  results = model_cap_more_train(three_channel_image)
  annotated_image = results[0].plot()
  bbox_data = []
  for r in results:
      boxes = r.boxes
      leftest = 0
      for box in boxes:
          b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
          c = box.cls  # get class index
          confidences = box.conf  # get confidence
          bbox_data.append((b, model_cap_more_train.names[int(c)], confidences))
  full_text = ""
  for bbox in bbox_data:
      full_text += f"{bbox[1]} "
  sorted_bbox_data = sorted(bbox_data, key=lambda x: x[0][0])
  threshold = 150
  threshold = int(threshold * annotated_image.shape[0] / 500)
  line_thickness = 2
  line_color = (255, 0, 0)
  line_y = threshold
  top_box = []
  bottom_box = []
  for bbox in sorted_bbox_data:
      if bbox[0][1] < threshold:
          top_box.append(bbox)
      else:
          bottom_box.append(bbox)

  top_text = ""
  bottom_text = ""
  for i, bbox in enumerate(top_box):
      top_text += bbox[1]
  for bbox in bottom_box:
      print(bbox)
      bottom_text += bbox[1]
  conf_threshold = 0.61
  top_box = []
  bottom_box = []
  top_text = ""
  bottom_text = ""
  count = 0
  for i,bbox in enumerate(sorted_bbox_data):
      coords, char, conf = bbox
      if conf < conf_threshold:

          # get new image from cropped bbox
          inconf_image = three_channel_image[
              int(bbox[0][1]) : int(bbox[0][3]), int(bbox[0][0]) : int(bbox[0][2])
          ]
          dilate = cv2.dilate(inconf_image, (3, 3), iterations=7)
          closed = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, (3, 3))
          fixed = reader.readtext(closed)

          _, etext, score = fixed[0] if len(fixed) > 0 else ("", "", 0)

          char = etext if len(etext) > 0 else char
          count += 1

      bbox = (bbox[0], char, bbox[2])

      if bbox[0][1] < threshold:
          top_box.append(bbox)
      else:
          bottom_box.append(bbox)
          
  def format_export_ver(ocr_result):
    res = ocr_result
    string_id = ""
    formatted_result = ""
    if "NSX" in ocr_result:
      string_id = "NSX"
      res = "NSX" + ocr_result.replace("N","").replace("S","").replace("X","").replace(" ","")
      formatted_result = res.split("X")[1]
    elif "HSD" in ocr_result:
      string_id = "HSD"
      res = "HSD" + ocr_result.replace("H","").replace("S","").replace("D","").replace(" ","")
      formatted_result = res.split("D")[1]
    else:
      return ocr_result
    formatted_result = "/".join(formatted_result[i : i + 2] for i in range(0, 6, 2))
    
    final_res = f"{string_id} {formatted_result}"
    return final_res 
    
  def format_lot_no(ocr_result):
      # if first layer is not number
      if not ocr_result[0].isdigit():
          return format_export_ver(ocr_result)
      ocr_result = "".join(ocr_result)

      # Remove special characters (in this case, ".")
      ocr_result = ocr_result.replace(" ", "").replace(".", "")

      # Add a dot after every 2 characters for the first 6 characters (21.03.25)
      formatted_result = ".".join(ocr_result[i : i + 2] for i in range(0, 6, 2))

      # Concatenate the remaining part, replacing 'K1' with 'K2'
      formatted_result = f"{formatted_result} {ocr_result[6:8]} {ocr_result[8:]}"
      return formatted_result

  top_text = ""
  bottom_text = ""

  pos_tolerance = 2
  for i, bbox in enumerate(top_box):
      current_box = bbox[0]
      current_conf = bbox[2]

      if i + 1 < len(top_box):
          next_box = top_box[i + 1][0]
          next_conf = top_box[i + 1][2]
          is_stacked_next = (next_box[0] - pos_tolerance < current_box[0] < next_box[0] + pos_tolerance)
          
          if is_stacked_next:
              if current_conf < next_conf:
                  continue

      if i - 1 >= 0:
          prev_box = top_box[i - 1][0]
          prev_conf = top_box[i - 1][2]
          is_stacked_before = (prev_box[0] - pos_tolerance < current_box[0] < prev_box[0] + pos_tolerance)
          
          if is_stacked_before:
              if current_conf < prev_conf:
                  continue
      top_text += bbox[1]

  for i, bbox in enumerate(bottom_box):
    current_box = bbox[0]
    current_conf = bbox[2]

    if i + 1 < len(bottom_box):
        next_box = bottom_box[i + 1][0]
        next_conf = bottom_box[i + 1][2]
        is_stacked_next = (next_box[0] - pos_tolerance < current_box[0] < next_box[0] + pos_tolerance)
        
        if is_stacked_next:
            if current_conf < next_conf:
                continue

    if i - 1 >= 0:
        prev_box = bottom_box[i - 1][0]
        prev_conf = bottom_box[i - 1][2]
        is_stacked_before = (prev_box[0] - pos_tolerance < current_box[0] < prev_box[0] + pos_tolerance)
        
        if is_stacked_before:
            if current_conf < prev_conf:
                continue
    bottom_text += bbox[1]


  print(f"**Final Output**")
  print(f"Top Text: ",format_lot_no(top_text))
  print(f"Bottom Text: ",format_export_ver(bottom_text))

def process_box(path:str):
  img = cv2.imread(path)
  model_cap_lot_number_full = YOLO("../runs/detect/box_localization/weights/last.pt")
  image_cap_lot_number_full = img
  rgb_image = cv2.cvtColor(image_cap_lot_number_full, cv2.COLOR_BGR2RGB)
  fresh_image = rgb_image.copy()
  results = model_cap_lot_number_full(rgb_image)
  annotated_image = results[0].plot()
  bbox_data = []
  cropped_image = []
  for r in results:
      boxes = r.boxes
      leftest = 0
      for box in boxes:
          b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
          c = box.cls  # get class index
          confidences = box.conf  # get confidence
          bbox_data.append((b, model_cap_lot_number_full.names[int(c)], confidences))
  for bbox in bbox_data:
      cropped_image = fresh_image[
          int(bbox[0][1]) : int(bbox[0][3]), int(bbox[0][0]) : int(bbox[0][2])
      ]
  results = reader.readtext(cropped_image, width_ths=0.7, link_threshold=0.8, decoder="greedy")
  top_bound = 0.3
  top_text = ""
  bottom_text = ""
  for (bbox, text, prob) in results:
      print("BBOX: ", bbox)
      print("crop: ", cropped_image.shape[0] * top_bound)
      print("shape: ", cropped_image.shape)
      if bbox[1][1] < cropped_image.shape[0] * top_bound:
          top_text += text
      else:
          bottom_text += text    
      cv2.line(cropped_image, (0, int(cropped_image.shape[0] * top_bound)), (cropped_image.shape[1], int(cropped_image.shape[0] * top_bound)), (255, 0, 0), 2)
      cv2.rectangle(cropped_image, bbox[0], bbox[2], (0, 255, 0), 2)
      cv2.putText(cropped_image, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  def sanitize_text(text: str):
      # text to uppercase
      text = text.upper()
      # switch I to 1 and S or Z to 2
      text = text.replace("I", "1").replace("S", "2").replace("Z", "2")
      # only show alphabet and number
      text = "".join([c for c in text if c.isalnum()])
      return text  
  def modify_text(text:str):
      # split "K"
      text = text.split("K")
      date_text: str = text[0]
      # give "." every 2 characters
      date_text = ".".join(date_text[i:i+2] for i in range(0, 6, 2))
      final_text = f"{date_text} K{text[1]}"
      return final_text  
  
  print(f"**Final Output**")
  print(f"Top Text: ",modify_text(sanitize_text(top_text)))
  print(f"Bottom Text: ",sanitize_text(bottom_text))


def process_soyjoy(path:str):
    def localize_image():
        # Load YOLOv8 model
        model = YOLO('./runs/detect/combined_final_lot_no/weights/best.pt')  # Replace with your model path
        count = 0
        show_per_row = 5
        # Iterate over all JPEG files in the directory
        filename = path
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            # Construct full file path
            file_path = filename
            # Read image
            image = cv2.imread(file_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fresh_image = rgb_image.copy()
            if image is None:
                print(f"Error reading image {file_path}")

            # Perform inference
            results = model(image)

            # Render results on the image
            annotated_image = results[0].plot()  # Adjust according to YOLOv8 results format
            bbox_data = []
            cropped_image = []
            for r in results:
                boxes = r.boxes
                leftest = 0
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls  # get class index
                    confidences = box.conf  # get confidence
                    bbox_data.append((b, model.names[int(c)], confidences))
            for bbox in bbox_data:
                cropped_image = fresh_image[
                    int(bbox[0][1]) : int(bbox[0][3]), int(bbox[0][0]) : int(bbox[0][2])
                ]
                cropped_image = cv2.resize(cropped_image, (250,100))
                
        return cropped_image
            
    def remove_special_character(text):
        return re.sub(r'[^a-zA-Z0-9]', '', text)

    def give_punctuations(text):
        # give "." each 2 characters
        return ".".join([text[i:i+2] for i in range(0, len(text), 2)])

    def remove_middle_character(text):
        # remove index 3 of text
        if len(text) < 5:
            return text
        return text[:2] + text[3:]
    
    image = localize_image()
    result = reader.readtext(image, text_threshold=0.6, width_ths=0.5, link_threshold=0.8)
    text_top = ""
    text_bottom = ""
    top_bound = 25
    for (bbox, text, prob) in result:
        # Ensure bbox coordinates are tuples
        top_left = tuple(map(int, bbox[0]))  # Convert to (x, y)
        bottom_right = tuple(map(int, bbox[2]))  # Convert to (x, y)
        
        if top_left[1] < top_bound:
            text_top += text
        else:
            text_bottom += text
        
        # Draw line to separate top and bottom text
        cv2.line(image, (0, top_bound), (image.shape[1], top_bound), (255, 0, 0), 2)
        
        # Draw rectangle
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        
        # Add text
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    
    print("-"*20)
    print(f"Top Text: {give_punctuations( remove_special_character(text_top))}")
    print(f"Bottom Text: {give_punctuations(remove_middle_character(remove_special_character(text_bottom)))}")
    print("-"*20)
         

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Script to load a model and process a file."
    )

    # Add the --path argument
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help="Path to the input file or directory"
    )

    # Add the --model argument
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['cap', 'box', 'soyjoy'],
        help="Specify the model to use (cap, box, or soyjoy)"
    )

    # Parse arguments
    args = parser.parse_args()
    
    if args.model == 'cap':
        process_cap(args.path)
    elif args.model == 'box':
        process_box(args.path)
    elif args.model == 'soyjoy':
        process_soyjoy(args.path)
    else:
        print(f"Model {args.model} not recognized.")
        return

    # Use the arguments
    print(f"Processing file(s) at: {args.path}")
    print(f"Using model: {args.model}")

if __name__ == "__main__":
    main()

