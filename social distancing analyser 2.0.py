from itertools import count
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time
import cv2
import numpy as np
from os import name, write
import dlib
import pickle
import datetime

#"ส่วนของการเรียกโมเดลและประกาศสตรัคเจอร์ต่างๆ"
############################################################################################################
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC,FACE_NAME = pickle.load(open('trainset.pk','rb'))
imgpath = "C:/Users/Admin/miniconda3/envs/Primlata/Social-Distancing-main/screenshot/"

countpic = 0
confid = 0.5
thresh = 0.5
# vname=""
vname=input("Video name in videos folder:  ")
if(vname==""):
   vname="Town.mp4"
#vid_path = "./videos/"+vname
vid_path = vname
angle_factor = 0.8
H_zoom_factor = 1.2
# Calibration needed for each video
#vs = cv2.VideoCapture(vid_path)
vs = cv2.VideoCapture(0)  ## USe this if you want to use webcam feed
frameRate = vs.get(5)
cur_time = time.time()
start_time_24h = cur_time
start_time_5sec = cur_time - 10

print(datetime)
##################################################################################################################


def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def T2S(T):
    S = abs(T/((1+T**2)**0.5))
    return S

def T2C(T):
    C = abs(1/((1+T**2)**0.5))
    return C

def isclose(p1,p2):

    c_d = dist(p1[2], p2[2])
    if(p1[1]<p2[1]):
        a_w = p1[0]
        a_h = p1[1]
    else:
        a_w = p2[0]
        a_h = p2[1]

    T = 0
    try:
        T=(p2[2][1]-p1[2][1])/(p2[2][0]-p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    S = T2S(T)
    C = T2C(T)
    d_hor = C*c_d
    d_ver = S*c_d
    vc_calib_hor = a_w*1.3
    vc_calib_ver = a_h*0.4*angle_factor
    c_calib_hor = a_w *1.7
    c_calib_ver = a_h*0.2*angle_factor
    
    print(p1[2], p2[2],(vc_calib_hor,d_hor),(vc_calib_ver,d_ver))
    if (0<d_hor<vc_calib_hor and 0<d_ver<vc_calib_ver):
        return 1
    elif 0<d_hor<c_calib_hor and 0<d_ver<c_calib_ver:
        return 2
    else:
        return 0


labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)

weightsPath = "./yolov3.weights"
configPath = "./yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
FR=0

writer = None
(W, H) = (None, None)

fl = 0
q = 0

#tydef
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        
    return (locs, preds)
    
#end tydef
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

while True:
    Date = datetime.datetime.now()
    (grabbed, frame ) = vs.read()
    ret, frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #กรอบตรงหน้า
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    cur_time = time.time() # Get current time เวลาปัจจุบัน
    elapsed_time_1min = cur_time - start_time_5sec
    
    
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        FW=W
        if(W<1075):
            FW = 1075
        FR = np.zeros((H+210,FW,3), np.uint8)

        col = (255,255,255)#สีพื้นหลังเฟรมใหญ่
        FH = H + 210
    FR[:] = col

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "person":

                if confidence > confid:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    if len(idxs) > 0:

        status = []
        idf = idxs.flatten()
        close_pair = []
        s_close_pair = []
        center = []
        co_info = []
        maskcount = []

        for i in idf:
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cen = [int(x + w / 2), int(y + h / 2)]
            center.append(cen)
            cv2.circle(frame, tuple(cen),1,(0,0,0),1)
            co_info.append([w, h, cen])

            status.append(0)
        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(co_info[i],co_info[j])

                if g == 1:

                    close_pair.append([center[i], center[j]])
                    status[i] = 1 #แดง
                    status[j] = 1
                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2
             


        total_p = len(center)
        low_risk_p = status.count(2)
        high_risk_p = status.count(1)
        safe_p = status.count(0)
        kk = 0
        Total_max = low_risk_p + high_risk_p +safe_p 


        for i in idf:
            
    
            cv2.line(FR,(0,H+1),(FW,H+1),(0,0,0),2)
            cv2.putText(FR, "Maximum in room :  " + str(Total_max), (10, H+60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(FR, "DateTime :  " + str(Date) , (380, H+60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.rectangle(FR, (20, H+80), (510, H+180), (100, 100, 100), 2)
            cv2.putText(FR, "Connecting lines shows closeness among people. ", (30, H+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 0), 2)
            cv2.putText(FR, "-- YELLOW: CLOSE", (50, H+90+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 170, 170), 2)
            cv2.putText(FR, "--    RED: VERY CLOSE", (50, H+40+110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.rectangle(FR, (535, H+80), (1060, H+140+40), (100, 100, 100), 2)
            cv2.putText(FR, "Bounding box shows the level of risk to the person.", (545, H+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 0), 2)
            cv2.putText(FR, "-- DARK RED: HIGH RISK", (565, H+90+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 2)
            cv2.putText(FR, "--   ORANGE: LOW RISK", (565, H+150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 2)

            cv2.putText(FR, "--    GREEN: SAFE", (565, H+170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

            
            tot_str = "People in room: " + str(total_p)
            high_str = "HIGH RISK COUNT: " + str(high_risk_p)
            low_str = "LOW RISK COUNT: " + str(low_risk_p)
            safe_str = "SAFE COUNT: " + str(safe_p)
            #maskcount_str = "PP NO Mask" +str(maskcount)
        
            cv2.putText(FR, tot_str, (10, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(FR, safe_str, (200, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 170, 0), 2)
            cv2.putText(FR, low_str, (380, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
            cv2.putText(FR, high_str, (630, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            (startx, starty) = (boxes[i][0], boxes[i][1])
            (endx, endy) = (boxes[i][2], boxes[i][3])

            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


            if status[kk] == 1:#ถ้าสถานะเป็นสีแดง ใกล้กัน
                
                if elapsed_time_1min >= 10 : #เริ่มจับเวลา
                    cv2.putText(frame,label,(startx, starty - 5), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2)
                    cv2.rectangle(frame,(startx, starty), (startx + endx , starty + endy), (0, 0, 150), 2)
                    start_time_5sec = cur_time #เคลียร์เวลาให้เป็นเวลาปัจจุบันอันใหม่
                    filename = imgpath + str(Date.strftime("%d-%m-%Y-%S")) + ".png" #เก็บบันทึกรูป
                    cv2.imwrite(filename,frame)#เก็บภาพจากเฟรม
                    countpic+=0.5
                    print(countpic)
                cv2.putText(frame,label,(x, y - 5), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2)
                cv2.rectangle(frame,(x, y), (x + w, y + h), (0, 0, 150), 2)
                
            elif status[kk] == 0: #ไกลกัน เขียว

                cv2.putText(frame,label,(x, y - 5), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2)   
                cv2.rectangle(frame,(x, y), (x + w, y + h), (0, 255, 0), 2)
                                                     
               
            else:
                cv2.putText(frame,label,(x, y - 5), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2)   
                cv2.rectangle(frame,(x, y), (x + w, y + h), (0, 120, 255), 2)                           
                   
            kk += 1
                  
        
        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        for b in s_close_pair:
            cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)
        
        FR[0:H, 0:W] = frame
        frame = FR
        for (x, y, w, h) in faces: #ส่วนกรอบที่ครอบหน้า
            img = frame[y - 10:y + h + 10, x - 10:x + w + 10][:, :, ::-1] #กรอบที่อยู่รอบหน้า
            dets = detector(img, 1)

            for k, d in enumerate(dets):
                
                shape = sp(img, d)
                face_desc0 = model.compute_face_descriptor(img, shape, 1)
                d = [] #เริ่มตรงนี้คือการนำรูปมาเทียบหน้าในวิดีโอ
                for face_desc in FACE_DESC:
                    d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))
                d = np.array(d)
                idx = np.argmin(d)
                if d[idx] < 0.5:
                    name = FACE_NAME[idx]
                    print(name)
                    cv2.putText(frame,name,(startX, startY - 5), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2)
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    #cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        #cv2.waitKey(1)#by Aiบ้านๆ

        cv2.imshow('Social distancing analyser', frame)
        cv2.waitKey(1)
        elapsed_time_24h = time.time() - start_time_24h
    
    

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        
        writer = cv2.VideoWriter("op_"+vname, fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)
  

    writer.write(frame)
#print("Processing finished: open"+"op_"+vname)
writer.release()
vs.release()