import cv2
import time
from datetime import datetime


cap = cv2.VideoCapture(0)
frameRate = cap.get(2) #frame rate
cur_time = time.time() # Get current time รับเวลาปัจจุบันมา
# start_time_24h measures 24 hours ให้เวลาปัจจุบันทั้งหมดมีครบ 24 ชั่วโมง
start_time_24h = cur_time
# start_time_1min measures 5 sec 
start_time_5sec = cur_time - 5 # Subtract 5 seconds for start grabbing first frame after one second (instead of waiting a minute for the first frame).
#นับเวลาถอยหลัง5วิเพื่อจับเฟรมแรก
imgpath = "C:/Users/Admin/miniconda3/envs/Primlata/Social-Distancing-Analyser-COVID-19-master/screenshot/"
#date = datetime.datetime.now()

while(True): #ถ้าเฟรมวิดีโอขึ้นลูปถึงจะทำงานเป็นทรู

    ret, frame = cap.read()
    cur_time = time.time() # Get current time เวลาปัจจุบัน
    elapsed_time_1min = cur_time - start_time_5sec # Time elapsed from previous image saving. เวลาถัดไปจากการบันทึกภาพก่อนหน้า เวลาปัจจุบัน - จับ 5 วิ
    #If 60 seconds were passed, reset timer, and store image. หากถึง 5 วิ ให้รีเซ็ตตัวจับเวลาแล้วจัดเก็บรูป
    if elapsed_time_1min >= 5 :  #หากถึง 5 วิ หรือมากกว่า
        start_time_5sec = cur_time #เคลียร์เวลาให้เป็นเวลาปัจจุบันอันใหม่
        filename = imgpath + str(datetime.now().strftime("%d-%m-%Y")) + ".png" #เก็บบันทึกรูป
        cv2.imwrite(filename,frame)
    
    # Show frame for testing #ส่วนของการแสดงแฟรมภาพวีดีโอ
    cv2.imshow('frame',frame)
    cv2.waitKey(1)
    elapsed_time_24h = time.time() - start_time_24h

cap.release()
cv2.destroyAllWindows()

