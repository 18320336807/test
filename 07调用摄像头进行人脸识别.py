#导入图像处理cv2工具
#pip install opencv-python   下载，其实就是opencv的python的接口
import  cv2
#导入人脸识别特征
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#导入眼睛识别特征
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
#调用摄像头
cap = cv2.VideoCapture(0)

while (True):
    #获取摄像头拍摄的画面   这个方法把帧读入进来赋值给frame等价于img  ret 表示成功与否  成功表示True
    ret,frame=cap.read()
    print(ret)
    #进行人脸识别返回坐标  三个参数   识别的人脸画面  放大比例  重复识别次数
    faces=face_cascade.detectMultiScale(frame,1.3,20)
    #打印人脸坐标
    #print(frame.shape)
    img=frame
    for (x, y, w, h) in faces:
        #画出人脸矩形
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #框选出人脸区域，人脸区域而不是全图中进行人脸检测，节省资源
        face_area=img[y:y+h,x:x+w]
        #眼睛识别返回眼睛坐标
        eyes=eye_cascade.detectMultiScale(face_area,1.3,10)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
    cv2.imshow('frame2',img)
    if cv2.waitKey(5) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
