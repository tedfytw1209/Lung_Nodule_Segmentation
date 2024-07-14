import gradio as gr
import cv2
import requests
import os
import torch
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox

# Example URLs for downloading images
file_urls = [
    "https://www.dropbox.com/scl/fi/n3bs5xnl2kanqmwv483k3/1_jpg.rf.4a59a63d0a7339d280dd18ef3c2e675a.jpg?rlkey=4n9dnls1byb4wm54ycxzx3ovi&st=ue5xv8yx&dl=0",
    "https://www.dropbox.com/scl/fi/asrmao4b4fpsrhqex8kog/2_jpg.rf.b87583d95aa220d4b7b532ae1948e7b7.jpg?rlkey=jkmux5jjy8euzhxizupdmpesb&st=v3ld14tx&dl=0",
    "https://www.dropbox.com/scl/fi/fi0e8zxqqy06asnu0robz/3_jpg.rf.d2932cce7e88c2675e300ececf9f1b82.jpg?rlkey=hfdqwxkxetabe38ukzbb39pl5&st=ga1uouhj&dl=0",
    "https://www.dropbox.com/scl/fi/ruobyat1ld1c33ch5yjpv/4_jpg.rf.3395c50b4db0ec0ed3448276965b2459.jpg?rlkey=j1m4qa0pmdh3rlr344v82u3am&st=lex8h3qi&dl=0",
    "https://www.dropbox.com/scl/fi/ok3izk4jj1pg6psxja3aj/5_jpg.rf.62f3dc64b6c894fbb165d8f6e2ee1382.jpg?rlkey=euu16z8fd8u8za4aflvu5qg4v&st=pwno39nc&dl=0",
    "https://www.dropbox.com/scl/fi/8r1fpwxkwq7c2i6ky6qv5/10_jpg.rf.c1785c33dd3552e860bf043c2fd0a379.jpg?rlkey=fcw41ppgzu0ao7xo6ijbpdi4c&st=to2udvxb&dl=0",
    "https://www.dropbox.com/scl/fi/ihiid7hbz1vvaoqrstwa5/7_jpg.rf.dfc30f9dc198cf6697d9023ac076e822.jpg?rlkey=yh67p4ex52wn9t0bfw0jr77ef&st=02qw80xa&dl=0",
]

def download_file(url, save_name):
    """Downloads a file from a URL."""
    if not os.path.exists(save_name):
        file = requests.get(url)
        with open(save_name, 'wb') as f:
            f.write(file.content)

# Download images
for i, url in enumerate(file_urls):
    download_file(url, f"image_{i}.jpg")

# Load YOLOv5 model (placeholder)
model_path = "best.pt"  # Path to your YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = attempt_load(model_path, device=device)  # Placeholder for model loading
model.eval()  # Set the model to evaluation mode

def preprocess_image(image_path):
    img0 = cv2.imread(image_path)
    img = letterbox(img0, 640, stride=32, auto=True)[0]  # Resize and pad to 640x640
    img = img.transpose(2, 0, 1)[::-1]  # Convert BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, img0

def infer(model, img):
    with torch.no_grad():
        pred = model(img)[0]
    return pred

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4].clip_(min=0, max=img1_shape[0])  # clip boxes
    return coords

def postprocess(pred, img0_shape, img):
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    results = []
    for det in pred:  # detections per image
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0_shape).round()
            for *xyxy, conf, cls in reversed(det):
                results.append((xyxy, conf, cls))
    return results

def detect_objects(image_path):
    img, img0 = preprocess_image(image_path)
    pred = infer(model, img)
    results = postprocess(pred, img0.shape, img)
    return results

def draw_bounding_boxes(img, results):
    for (x1, y1, x2, y2), conf, cls in results:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return img

def show_preds_image(filepath):
    results = detect_objects(filepath)
    img0 = cv2.imread(filepath)
    img_with_boxes = draw_bounding_boxes(img0, results)
    return cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

# Define Gradio components
input_component = gr.components.Image(type="filepath", label="Input Image")
output_component = gr.components.Image(type="numpy", label="Output Image")

# Create Gradio interface
interface = gr.Interface(
    fn=show_preds_image,
    inputs=input_component,
    outputs=output_component,
    title="Lung Nodule Detection [ Segmentation Model ]",
    examples=[
        "image_1.jpg",
        "image_2.jpg",
        "image_3.jpg",
        "image_4.jpg",
        "image_5.jpg",
        "image_6.jpg",
    ],
    description=' "This online deployment proves the effectiveness and efficient function of the machine learning model in identifying lung cancer nodules. The implementation of YOLO for core detection tasks is employed that is an efficient and accurate algorithm for object detection. Through the precise hyper-parameter tuning process, the model proposed in this paper has given an impressive boost in the performance. Moreover, the model uses Retinanet algorithm which is recognized as the powerful tool effective in dense object detection. In an attempt to enhance the model’s performance, the backbone of this architecture consists of a Feature Pyramid Network (FPN). The FPN plays an important role in boosting the model’s capacity in recognizing objects in different scales through the construction of high semantic feature map in different resolutions. In conclusion, this deployment encompasses YOLOv5, hyperparameter optimization, Retinanet, and FPN as one of the most effective and modern solutions for the detection of lung cancer nodules." ~ Basil Shaji 😇',
    live=False,
)

interface.launch()
