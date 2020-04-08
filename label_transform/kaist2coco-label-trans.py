import numpy as np
import cv2
import os
import sys

#set your data_set absolute path
#as for me, for example 
# test example
kaist_img_path ='picture/'
kaist_label_path = 'temp/'



#transformed lables to save path
kaist_label_tosave_path = 'output/'
# kaist_label_tosave_path = '/home/dlsj/Documents/PyTorch-YOLOv3/data/extracted_labels'
#the absolute ptah of your data set
#kaist_data_real_path = '/home/pakcy/Desktop/PyTorch-YOLOv3/data/kaist/images/train/'

index = 0
cvfont = cv2.FONT_HERSHEY_SIMPLEX
'''
0 = person, 1 = people, 2 = cyclist
'''
kaist_names = open('kaist.names','r')
kaist_names_contents = kaist_names.readlines()                
kaist_images = os.listdir(kaist_img_path)
kaist_labels = os.listdir(kaist_label_path)

kaist_images.sort()
kaist_labels.sort()

kaist_names_dic_key = []
for class_name in kaist_names_contents:
    kaist_names_dic_key.append(class_name.rstrip())
values = range(len(kaist_names_dic_key))
kaist_names_num = dict(zip(kaist_names_dic_key,values))

# print(kaist_names_num)
# print(kaist_labels)

#Image sets directory
# f = open('train.txt','w')
# for img in kaist_images:
#     f.write(kaist_img_path+img+'\n')
# f.close()

#kaist数据集 相对坐标 转换为绝对坐标
for indexi in range(len(kaist_images)):
    kaist_img_totest_path = kaist_img_path + kaist_images[indexi]
    kaist_label_totest_path = kaist_label_path + kaist_labels[indexi]
    # print(kaist_img_totest_path, kaist_label_totest_path)
    
    kaist_img_totest = cv2.imread(kaist_img_totest_path)
    # print(kaist_img_totest,type(kaist_img_totest))
    img_height, img_width = kaist_img_totest.shape[0],kaist_img_totest.shape[1]
    # print(img_height, img_width)
    
    kaist_label_totest = open(kaist_label_totest_path,'r')
    
    label_contents = kaist_label_totest.readlines()
    # print(label_contents)
    real_label = open(kaist_label_tosave_path + kaist_labels[indexi],'w')
    
    for line in label_contents:
        if (line=='% bbGt version=3\n'):
        	continue #% bbGt version =3\n skip this part the first line
        data = line.split(' ') 
        print(data)
        x=y=w=h=0
        if(len(data) == 12):
            class_str = data[0]
            if(class_str != 'cyclist'): #ignore cyclist first
                # for kaist calls is a string
                # trans this to number by using kaist.names
                #(x,y) center (w,h) size
                x1 = float(data[1])
                y1 = float(data[2])
                x2 = float(data[3])
                y2 = float(data[4])
                
                # intx1 = int(x1)
                # inty1 = int(y1)
                # intx2 = int(x2)
                # inty2 = int(y2)

                bbox_center_x = float( x1  / img_width)
                bbox_center_y = float( y1  / img_height)
                bbox_width = float(x2 / img_width)
                bbox_height = float(y2 / img_height)
                # print(bbox_center_x, bbox_center_y, bbox_width, bbox_height)

                #print(kaist_names_contents[class_num])
                # cv2.putText()
                # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
                #cv2.putText(kaist_img_totest, class_str, (intx1, inty1+3), cvfont, 2, (0,0,255), 1)
                # cv2.rectangle()
                # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
                #cv2.rectangle(kaist_img_totest, (intx1,inty1), (intx2,inty2), (0,255,0), 2)
                line_to_write = str(kaist_names_num[class_str]) + ' ' + str(bbox_center_x)+ ' ' + str(bbox_center_y)+ ' ' + str(bbox_width)+ ' ' + str(bbox_height) +'\n' #1 line of all texts
                real_label.write(line_to_write) #write to the new file
                sys.stdout.write(str(int((indexi/len(kaist_images))*100))+'% '+'*******************->' "\r" )
                sys.stdout.flush()

    # cv2.imshow(str(indexi)+' kaist_label_show',kaist_img_totest)    
    # cv2.waitKey(0)
    real_label.close()
kaist_names.close()
print("Labels transform finished!")
