import numpy as np
import colorsys
from PIL import Image
import cv2 as cv
#没用什么复杂的算法,很遗憾没用矩阵操作

#这是一个制造抖动的程序-单列
def test_err_1(x,rand_x): 
    res=[]
    for i,i_x in zip(x,rand_x):
        if i>i_x:
            res.append(i)
        else:
            res.append(abs(i-i_x)) #做一点尝试
    return res
def test_err_2(x,rand_x): 
    res=[]
    for i,i_x in zip(x,rand_x):
        res_res=[]
        for j,j_x in zip(i,i_x):
            if j>j_x:
                res_res.append(j)
            else:
                res_res.append(abs(j-j_x))
                #res_res.append(0)
        res.append(res_res)
    return np.array(res)
def test_err_2_np(x,rand_x): #numpy版本 -- 很快
    #x[x<=rand_x]=np.abs(x[x<=rand_x]-rand_x[x<=rand_x])
    x[x<=rand_x]=0 #回头看看有什么比较好的方法优化
    return x
def test_err_2_per_rand(x): #这样搞很有显示屏的花屏感
    rand_x=np.random.randint(0,255,[cap_size[1],cap_size[0]])
    x[x<=rand_x]=0
    return x
test_img_path="E:/py_project/halftone/asset/人像.png"
test_v_path="E:/py_project/halftone/asset/素材3.mp4"
def test_err_fiffusion(x,rank):  #可以到达一个诡异的效果，或者可以用于怪核
    #转换的级别
    #转换的角度 - 给指定方向加上去 sin:1:cos
    angle=30
    #rank=50 #上限是255
    r,c=x.shape[0],x.shape[1]
    for i in range(r):
        for j in range(c):
            temp=x[i,j] / rank
            err= temp-int(temp)
            err_base=err*rank #扩散的基础值
            x[i,j]=int(temp)
            if(j+1<c):
                x[i,j+1]+=err_base/3
                if(i+1<r):
                    x[i+1,j+1]+=err_base/3
            if(i+1<r):
                x[i+1,j]+=err_base/3
    return x*rank
cap = cv.VideoCapture(test_v_path)
cap_size=(int(cap.get(3)),int(cap.get(4))) #分辨率
fps=cap.get(5) #帧率
fourcc = cv.VideoWriter_fourcc(*'DIVX') #编码格式
out = cv.VideoWriter('output.mp4',fourcc,fps,cap_size) #同帧率，同分辨率，同编码格式
rand_v_x=np.random.randint(0,255,[cap_size[1],cap_size[0]]) #随机抖动矩阵
while(True):
    ret, frame = cap.read()
    if(ret==False):
        break
    gray_r,gray_g,gray_b = cv.split(frame)#获取通道
    #进行处理    
    gray_r=test_err_2_np(gray_r,rand_v_x)
    gray_g=test_err_2_np(gray_g,rand_v_x)
    gray_b=test_err_2_np(gray_b,rand_v_x)
    #gray_r=test_err_fiffusion(gray_r)
    #gray_g=test_err_fiffusion(gray_g)
    #gray_b=test_err_fiffusion(gray_b)
    gray=cv.merge([gray_r,gray_g,gray_b])
    out.write(gray) #写入视频
image = Image.open(test_img_path).convert('RGB')
image.load()
r_o, g_o, b_o= image.split() #拆分
image_arr=np.array(image)
r=image_arr[:,:,0]
g=image_arr[:,:,1]
b=image_arr[:,:,2]
#图片通道数据以下
test_img_sz=r.size # test:478,485
#rand_x=np.random.randint(0,255,test_img_sz[0]*test_img_sz[1]) #随机抖动矩阵-单列版本
print(f"R:{r}")
print(f"G:{g}")
print(f"B:{b}") 
#图片通道数据以上
#怪核序列帧

out_2 = cv.VideoWriter('output2.mp4',fourcc,40.0,(478,485)) #同帧率，同分辨率，同编码格式
#np.save("filename.npy",a)
for i in range(255):
    r_test=test_err_fiffusion(r.copy(),i+1)
    g_test=test_err_fiffusion(g.copy(),i+1)
    b_test=test_err_fiffusion(b.copy(),i+1)
    gray=cv.merge([r_test,g_test,b_test])
    out_2.write(gray) #写入视频
    print(i)
r=test_err_fiffusion(r,50)
g=test_err_fiffusion(g,50)
b=test_err_fiffusion(b,50)
image_arr=np.dstack((r,g,b)) #合并
arr2im = Image.fromarray(image_arr)
#组装
#r_o.putdata(test_err_1(r,rand_x))
#g_o.putdata(test_err_1(g,rand_x))
#b_o.putdata(test_err_1(b,rand_x))
#r_o.putdata(test_err_fiffusion(r))
#g_o.putdata(test_err_fiffusion(g))
#b_o.putdata(test_err_fiffusion(b))
#image = Image.merge('RGB', (r_o, g_o, b_o))
arr2im.save('output.png')
#image.save('output.png')
print("ending-s")