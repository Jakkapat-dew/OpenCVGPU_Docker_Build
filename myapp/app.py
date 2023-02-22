from flask import Flask, render_template, send_from_directory, request, jsonify, redirect, url_for
import json
import os
import base64
import cv2
import numpy as np
import time
import pandas as pd

app = Flask(__name__)

class dataJSON:
    def __init__(self):
        self.request = request
        self.json_data = self.request.get_json()

    def get_image(self, key="image"):
        image_string = self.json_data[key]
        image_bytes = base64.b64decode(image_string)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

        return image
    
    def get_string(self, key="minDist"):
        return self.json_data[key]
    
## ----- Fn ----------------------------------------------------------- ##
def draw_circle(imgBGR, circles):
    
    imgBGR = imgBGR.copy()
    # color conf
    outer_color = (0,255,0)
    center_color = (0,0,255)
    
    circles = np.uint16(np.around(circles))
    for i in circles:
        cx, cy, r = i[0], i[1], i[2]
        # draw the outer circle
        cv2.circle(imgBGR, (cx, cy), r, outer_color, 4)
        # draw the center of the circle
        cv2.circle(imgBGR, (cx, cy), 3, center_color, 4)
        
    return imgBGR

def houghCircle(image, gpuEnable, 
                dp, minDist, cannyThreshold, accThreshold, 
                minRadius, maxRadius):
    
    ## preprocess
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(src=gray, ksize=(9,9), sigmaX=1.5)

    if gpuEnable == 1:
        # load to gpu memory
        cu_img = cv2.cuda_GpuMat()
        cu_img.upload(blur)
        
        # create HoughCircles Detector
        cu_hough = cv2.cuda.createHoughCirclesDetector(dp=dp, minDist=minDist, 
                                                    cannyThreshold = cannyThreshold, votesThreshold=accThreshold,
                                                    minRadius=minRadius, maxRadius=maxRadius)
        
        gts = time.time()
        cuCircles = cu_hough.detect(cu_img)
        gte = time.time()
        calcTime = gte-gts
        print("GPU time: "'{0:.3f}s'.format(calcTime))

        # load result back to cpu memory
        gpuCircles = cuCircles.download()
        print("gpuCircles: ", gpuCircles)
        
        Circles = gpuCircles
        
    else:
        ## ----- CPU start ----- ##
        cts = time.time()
        cpuCircles = cv2.HoughCircles(image=blur, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, 
                                    param1=cannyThreshold, param2=accThreshold, minRadius=minRadius, maxRadius=maxRadius)
        cte = time.time()
        calcTime = cte-cts
        print("CPU time: "'{0:.3f}s'.format(calcTime))
        print("cpuCircles: ", cpuCircles)
        
        Circles = cpuCircles
    
    # --- To result --- #
    sorted_circles = [] #[x, y, r]

    if Circles is not None:
        # sorting and delete duplicate
        df = pd.DataFrame(Circles[0, :], columns=['x', 'y', 'r'])
        dfsort1 = df.sort_values(by=['x','y','r'], ascending=False)
        df_filtered = dfsort1.drop_duplicates(subset=['x', 'y'], keep='first', inplace=False)
        df_sorted = df_filtered.sort_values(by=['x','y','r'], ascending=False)
        print("df_sorted: ", df_sorted)
        sorted_circles = df_sorted.values.tolist()
        
    print("sorted_circles: ", sorted_circles)
    
    if len(sorted_circles) != 2:
        sorted_circles =[]

    return sorted_circles, np.round(calcTime, 3)




@app.route('/')
def index():
    print('Request for index page received')
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

@app.route("/detect_circle", methods=["POST"])
def detect_circle():

    ## hough circle parameter    
    data_json = dataJSON()
    input_image = data_json.get_image(key="image")
    dp = data_json.get_string(key="dp")
    minDist = data_json.get_string(key="minDist")
    cannyThreshold = data_json.get_string(key="cannyThreshold")
    accThreshold = data_json.get_string(key="accThreshold")
    minRadius = data_json.get_string(key="minRadius")
    maxRadius = data_json.get_string(key="maxRadius")
    GPUEnable = data_json.get_string(key="GPUEnable")
    
    # --- check if gou is not enable --- 
    if GPUEnable == "1" and cv2.cuda.getCudaEnabledDeviceCount() == 0:
        calcTime = 0
        dict_circles = None
        imghough = input_image.copy()
        strStatus = "OpenCV GPU is not enable"
    
    else:
        
        # perform hough circle by opencv gpu
        sorted_circles, calcTime = houghCircle(input_image.copy(), int(GPUEnable), 
                                            int(dp), int(minDist), int(cannyThreshold), int(accThreshold), 
                                            int(minRadius), int(maxRadius))
        
        # check hough circle is blank
        if len(sorted_circles) != 0:
            dict_circles = [{"x": x, "y": y, "r": r} for x, y, r in sorted_circles]
            imghough = draw_circle(imgBGR=input_image.copy(), circles=sorted_circles)
            strStatus = "circle is detected"
        else:
            dict_circles = None
            imghough = input_image.copy()
            strStatus = "circle is not detected"
        
    # convert numpy array to byte
    _, imgencode = cv2.imencode('.png', imghough)
    array_bytes = imgencode.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    
    # data pack
    data = {"status": strStatus,
            "calculationTime": calcTime,
            "circles": dict_circles,
            "imghough_b64": base64_string}
    
    return json.dumps(data)

# start server
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)