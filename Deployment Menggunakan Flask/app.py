# load libraries
from flask import Flask, render_template
from flask import request 
import torch as nn
import cv2 as cv 
import os
import torch
import torchvision


# define path and parent route
app = Flask(__name__, static_folder = 'static')
UPLOAD_FOLDER = ("K:\SMT 5\Studi Independen\Proyek Akhir\Proyek Akhir\Deployment Menggunakan Flask") 

# define model
model = nn.hub.load('ultralytics/yolov5', 'custom', path = "model/best.pt", force_reload = True)
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt', force_reload=True)

# helper function
def score_frame(frame):
    results = model([frame])
    labels, coordinate = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    return labels, coordinate

def class_to_label(x):
    return model.names[int(x)]

def plot_boxes(results, frame):
    labels, coordinate = results 
    n = len(labels)
    X_shape, y_shape = frame.shape[0], frame.shape[1]

    for i in range(n):
        row = coordinate[i]
        print(row)
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            color = (0, 255, 0)
            cv.rectangle(frame, (x1, y1), (x2, y2),  color, 1)
            cv.putText(frame, class_to_label(labels[i]), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 1)

    return frame 


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        image_file = request.files['image']
        if image_file: # condition True
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location) # save image

            # predict image
            img = cv.imread(image_location)
            # get result
            result = score_frame(img)
            frame = plot_boxes(result, img)
            
            # save image
            cv.imwrite('static/result.jpg', frame)
        return render_template('hasil.html')

    else:
        return render_template('index.html')

if __name__=='__main__':
    app.run(port=12000, debug=True)