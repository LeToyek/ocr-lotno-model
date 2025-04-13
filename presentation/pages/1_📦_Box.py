import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from streamlit_image_select import image_select
from ultralytics import YOLO
import cv2
import easyocr
from io import StringIO
import numpy as np
from PIL import Image

reader = easyocr.Reader(["en"])

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

with st.sidebar:
  st.write("## Navigation")
  st.subheader("1. [Data Crawling](#data_crawling)")
  st.subheader("2. [Data Annotating](#data_annotating)")
  st.subheader("3. [Data Filtering](#data_filtering)")
  st.subheader("4. [OCR](#ocr)")
  st.subheader("5. [Postprocessing](#postprocessing)")
  st.subheader("6. [Finalization](#finalization)")
  
    
# Title of the app
st.title("Hasil Riset OCR Lot Number pada Box Pocari")

# Adding some text
st.write("Riset ini bertujuan untuk mengetahui hasil dari OCR Lot Number pada Box")

# Adding a header
st.header("Hasil Riset")

# 1. Data Crawling
st.markdown('<a id="data_crawling"></a>', unsafe_allow_html=True)
st.subheader("1. Data Crawling")
st.write(
    "Data crawling dilakukan dengan cara mengambil data dari CAP botol Pocari yang diambil menggunakan kamera handphone"
)

image_folder = "../data/box"
images = []
for i, filename in enumerate(os.listdir(image_folder)):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
        # Construct full file path
        file_path = os.path.join(image_folder, filename)
        images.append(file_path)
        
# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

img = image_select("Foto Mentah dari HP", images)
if uploaded_file is not None:
    # Read image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.write(f"Error reading image {uploaded_file.name}")
    else:
        img = image

# write bold
st.write("**Gambar dipilih**")
st.image(img, use_column_width=True)

# 2. Anotating Bounding Box Lot No
st.markdown('<a id="data_annotating"></a>', unsafe_allow_html=True)
st.subheader("2. Annotating Bounding Box Lot No")
st.write(
    "Setelah data crawling dilakukan, selanjutnya adalah melakukan anotasi pada bounding box Lot Number, tujuannya adalah untuk memberi tahu mesin di mana letak Lot Number pada CAP botol Pocari"
)
st.write("**Hasil Anotasi Bounding Box Lot No**")
# italic
st.write(
    "*Silahkan pilih gambar di atas untuk melihat setiap foto yang terbounding box*"
)
model_cap_lot_number_full = YOLO("../runs/detect/box_localization/weights/last.pt")
image_cap_lot_number_full = img if uploaded_file is not None else cv2.imread(img)

rgb_image = cv2.cvtColor(image_cap_lot_number_full, cv2.COLOR_BGR2RGB)
fresh_image = rgb_image.copy()
results = model_cap_lot_number_full(rgb_image)
annotated_image = results[0].plot()
print("Annotated Image: ", annotated_image)
st.image(annotated_image)

st.write(
    "Gambar yang dipilih kemudian dipotong sesuai dengan bounding box yang telah ditentukan oleh model machine learning. Berikut adalah hasilnya"
)
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
    st.write(f"**Gambar Terpotong**")
    st.image(cropped_image)
    
st.markdown('<a id="data_filtering"></a>', unsafe_allow_html=True)
st.subheader("3. Penambahan Filter pada Gambar (Preprocessing)")
st.write(
    "Setelah gambar terpotong, selanjutnya adalah menambahkan filter pada gambar agar lebih mudah untuk dilakukan OCR. Hal ini dilakukan karena gambar yang terpotong masih memiliki banyak noise sehingga akan mempengaruhi hasil dari OCR"
)

gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
st.write(f"**A. Gambar Grayscale**")
st.write(
    "Mengubah gambar menjadi grayscale yaitu menjadikan gambar hanya memiliki satu channel saja"
)
st.image(gray_image)


kernel_size = 3
max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

# blur_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 15, 7, 21)
more_denoised = cv2.medianBlur(denoised_image, 15)

st.write(f"**B. Gambar Denoised**")
st.write(
    "Menghilangkan noise pada gambar dengan menggunakan teknik denoising"
)
st.image(denoised_image)

# Thresholding
thresh = cv2.adaptiveThreshold(
    more_denoised,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=29,
    C=2
)

st.write(f"**C. Gambar Threshold**")
st.write(
    "Mengubah gambar menjadi threshold dengan menggunakan metode adaptive thresholding"
)
st.image(thresh)

dilated = cv2.dilate(thresh, max_kernel, iterations=3)
st.write(f"**D. Gambar Dilated**")
st.write(
    "Mengubah gambar menjadi dilated dengan menggunakan kernel maksimum"
)
st.image(dilated)

closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, max_kernel)
st.write(f"**E. Gambar Closed**")
st.write(
    "Mengubah gambar menjadi closed dengan menggunakan kernel maksimum"
)
st.image(closed)

reader = easyocr.Reader(["en"])
st.write(f"**F. Hasil OCR**")
st.write(
    "Hasil dari OCR pada gambar yang telah di preprocessing"
)
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
st.image(cropped_image)
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

st.write("TOP TEXT ", modify_text(sanitize_text(top_text)))
st.write("BOTTOM TEXT ", sanitize_text(bottom_text))
    


# 4.OCR

st.markdown('<a id="ocr"></a>', unsafe_allow_html=True)
st.subheader("4. OCR")
st.write(
    "Setelah gambar telah dilakukan preprocessing, selanjutnya adalah melakukan OCR pada gambar tersebut"
)
model_box = YOLO(
    "../runs/detect/box_single/weights/best.pt"
)
three_channel_image = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
results = model_box(three_channel_image)
annotated_image = results[0].plot()
st.image(annotated_image)
st.write(
    "Hasil dari OCR sudah bagus, tetapi urutan pada teks yang diekstrak tidak urut dari kiri ke kanan dan belum terformat dengan baik. Berikut adalah hasil ekstrasi teks dari gambar yang telah di OCR"
)
bbox_data = []
for r in results:
    boxes = r.boxes
    leftest = 0
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls  # get class index
        confidences = box.conf  # get confidence
        bbox_data.append((b, model_box.names[int(c)], confidences))
full_text = ""
for bbox in bbox_data:
    full_text += f"{bbox[1]} "
st.write(f"{full_text}")

# 5. Postprocessing

st.markdown('<a id="postprocessing"></a>', unsafe_allow_html=True)
st.subheader("5. Postprocessing")
st.write(
    "Langkah ini ditujukan untuk memformat hasil ekstraksi teks dari OCR agar lebih mudah dibaca dan sesuai format"
)
st.write("**Pengurutan Bounding Box**")
st.write(
    "Bounding box yang telah di OCR diurutkan dari kiri ke kanan menggunakan code di bawah ini"
)
sorted_bbox_data = sorted(bbox_data, key=lambda x: x[0][0])
st.code(
    """
        sorted_bbox_data = sorted(bbox_data, key=lambda x: x[0][0])
"""
)
st.write("**Pemisahan Teks**")
st.write(
    "Teks dipisah menjadi dua bagian yaitu bagian atas dan bagian bawah. Caranya adalah menetapkan batas untuk memindahkan teks ke bagian atas dan bagian bawah"
)

# adjust the image_height with 500 then we can get the real threshold. If height = 500 then threshold = 150. adjust it with annotated image height
threshold = 150
threshold = int(threshold * annotated_image.shape[0] / 500)
line_thickness = 2
line_color = (255, 0, 0)
line_y = threshold
line_image = annotated_image.copy()
cv2.line(
    line_image, (0, line_y), (line_image.shape[1], line_y), line_color, line_thickness
)
st.image(line_image)

st.write("**Hasil Pemisahan Teks**")
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

st.write(f"**Bagian Atas**")
st.write(f"{top_text}")
st.write(f"**Bagian Bawah**")
st.write(f"{bottom_text}")

st.write(
    "Meskipun model telah dapat mengidentifikasi dengan baik, terkadang, hasil yang ditampilkan masih memiliki `confidences` yang rendah. Oleh karena itu, pada penelitian ini penulis menambahkan metode untuk mengidentifikasi kembali dengan model OCR yang sudah diimplementasikan di banyak project yaitu `EasyOCR`. Penulis menetapkan threshold sebesar 0.6 untuk mengidentifikasi kembali teks yang memiliki `confidences` rendah"
)

conf_threshold = 0.5
top_box = []
bottom_box = []
top_text = ""
bottom_text = ""
count = 0
for i,bbox in enumerate(sorted_bbox_data):
    coords, char, conf = bbox
    if conf < conf_threshold:
        st.divider()

        st.write(f"**Raw Char**")
        st.write(f"{char}")
        # get new image from cropped bbox
        inconf_image = three_channel_image[
            int(bbox[0][1]) : int(bbox[0][3]), int(bbox[0][0]) : int(bbox[0][2])
        ]
        st.write(f"**Inconf Image**")
        st.image(inconf_image)
        dilate = cv2.dilate(inconf_image, (3, 3), iterations=7)
        closed = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, (3, 3))
        fixed = reader.readtext(closed)

        _, etext, score = fixed[0] if len(fixed) > 0 else ("", "", 0)
        st.write(f"**Confidence**")
        st.write(f"yolo {conf}")
        st.write(f"easyocr {score}")

        char = etext if len(etext) > 0 and score > conf_threshold else char
        st.write(f"**Fixed Character**")
        st.write(f"{char}")
        count += 1

    bbox = (bbox[0], char, bbox[2])

    if bbox[0][1] < threshold:
        print("=============TOP BOX==============")
        print(bbox)
        top_box.append(bbox)
    else:
        print("=============BOX==============")
        print(bbox)
        bottom_box.append(bbox)

if count > 0:
    st.write(f"**Jumlah karakter yang di fix**")
    st.write(f"{count}")
else:
    st.write(f"**Tidak ada karakter yang di fix**")

# 6. Result

st.markdown('<a id="finalization"></a>', unsafe_allow_html=True)
st.subheader("6. Finalization")
st.write(
    "Langkah terakhir yang dilakukan adalah dengan melakukan formatting pada data yang telah diproses. Proses ini bertujuan untuk menyesuaikan hasil OCR dengan format pada Lot No yang telah ditentukan. Berikut adalah hasil akhir dari OCR Lot Number pada CAP Botol Pocari"
)

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

for bbox in bottom_box:
    bottom_text += bbox[1]

st.write(f"**Final Output**")
st.write(f"Top Text: `{format_lot_no(top_text)}`")
st.write(f"Bottom Text: `{format_export_ver(bottom_text)}`")

st.subheader("Kesimpulan")
st.write(
    "Dari hasil penelitian ini, penulis dapat menyimpulkan bahwa model machine learning yang telah diimplementasikan dapat mengidentifikasi dengan baik pada CAP botol Pocari. Namun, terkadang hasil yang ditampilkan masih memiliki `confidences` yang rendah. Oleh karena itu, penulis menambahkan metode untuk mengidentifikasi kembali dengan model OCR yang sudah diimplementasikan di banyak project yaitu `EasyOCR`. Penulis menetapkan threshold sebesar 0.6 untuk mengidentifikasi kembali teks yang memiliki `confidences` rendah. Dengan demikian, hasil yang ditampilkan sudah dapat dibaca dengan baik dan sesuai format yang telah ditentukan"
)

uploaded_file = None



