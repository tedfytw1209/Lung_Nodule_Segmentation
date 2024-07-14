import gradio as gr
import cv2
import requests
import os
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
import yolov5.utils.plots
import yolov5.utils.plots
dir(yolov5.utils.plots)

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
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name,'wb').write(file.content)

# Download files
for i, url in enumerate(file_urls):
    if "mp4" in url:
        download_file(url, "video.mp4")
    else:
        download_file(url, f"image_{i}.jpg")

# Load YOLOv5 model
model_path = "best.pt"
model = attempt_load(model_path, device=torch.device('cpu'))

def show_preds_image(image_path):
    img0 = cv2.imread(image_path)  # Open image

    # Inference
    results = model(img0)  # Pass image to model

    # Process detections
    for i, det in enumerate(results.pred[0]):
        # Draw bounding boxes
        plot_one_box(det.cpu().numpy(), img0, color=(0, 0, 255), line_thickness=2)

    return cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

inputs_image = [
    gr.inputs.Image(type="file", label="Input Image"),
]
outputs_image = [
    gr.outputs.Image(type="numpy", label="Output Image"),
]

interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="YOLOv5 Object Detection",
    examples=[["image_0.jpg"], ["image_1.jpg"]],
    live=False,
)

interface_image.launch()
