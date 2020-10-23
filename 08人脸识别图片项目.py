#导入图像处理cv2工具
#pip install opencv-python   下载，其实就是opencv的python的接口
import  cv2
#读入图片   通过cv2中的imread（）的方法读取在img这个变量里边
img=cv2.imread('33.jpg')
#设置图片的宽度
resize_img=cv2.resize(img,dsize=(400,550))
#导入人脸级联分类器引擎 命名为face_engine，按’.xml‘文件来匹配的人脸检测模型
face_engine=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
#用这个模型进行人脸识别，返回的faces 为所有 人脸坐标  列表  ，1.3为放大比例， 5为重复识别次数
faces=face_engine.detectMultiScale(resize_img,scaleFactor=1.3,minNeighbors=5)
#对人脸进行如下操作  右上角和左下角的坐标    画矩形
for (x,y,w,h) in faces:
    #画出人脸矩形调用rectangle，蓝色BGR色彩体系  画笔宽度为2
    cv2.rectangle(resize_img,(x,y),(x+w,y+h),(0,255,0),thickness=4)
#在 窗口展示效果  图片命名为img2
cv2.imshow('img2',resize_img)
#监听键盘上的任何按键，如有按键就退出窗口，释放内存，保存图片命名为
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('img2.jpg',resize_img)