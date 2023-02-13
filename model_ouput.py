import numpy as np
import colorsys
from PIL import Image
import cv2 as cv
import paddle


def img_cov(img_np): 
    res=[]
    for i in range(3):
        res.append(img_np[:,:,i])
    return np.array(res)

def cov_img(img_np):
    r=img_np[0]
    g=img_np[1]
    b=img_np[2]
    res=np.append(r[:,:,np.newaxis],g[:,:,np.newaxis],axis=2)
    res=np.append(res,b[:,:,np.newaxis],axis=2)
    return res

def output_img(img):
    img = img.astype(dtype='float32')
    img=img_cov(img)
    pred = model(img[np.newaxis,:,:,:]) 
    img=cov_img(pred[0])
    return img
model = paddle.jit.load("model/inference_model")
model.eval()
vedio_path="asset/#祝福.mp4"
cap = cv.VideoCapture(vedio_path)
cap_size=(int(cap.get(3)),int(cap.get(4))) #分辨率
fps=cap.get(5) #帧率
fourcc =cv.VideoWriter_fourcc("M","P","4","V")
out = cv.VideoWriter('output_for_model.mp4',fourcc,fps,cap_size) #同帧率，同分辨率，同编码格式
print(f"视频分辨率为{cap_size},帧率为{fps}")
while(True):
    ret, frame = cap.read()
    if(ret==False):
        break
    #gray_r,gray_g,gray_b = cv.split(frame)#获取通道
    #进行处理    
    #gray=np.array([gray_r,gray_g,gray_b])
    frame=np.array(frame,dtype="uint8")
    gray=output_img(frame)
    #gray=Image.fromarray(np.uint8(gray))
    gray=gray.astype("uint8")
    cv.imshow("2",gray)
    cv.waitKey(1)
    #gray=cv.merge([gray[0],gray[1],gray[2]])
    #print(frame.shape)
    #print(gray.shape)
    out.write(gray) #写入视频
    print("@",end="")
cap.release()
