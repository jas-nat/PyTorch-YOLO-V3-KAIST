import numpy as np
import cv2
import os
import sys
import time
# so far work only visible data
#comment for what kind of training
flag = 'day' 
flag = 'night' #uncomment this if you want train for night

#set your data_set absolute path
#set the path for training and test
kaist_img_path ='/home/dlsj/Documents/PyTorch-YOLOv3/data/kaist/images/'
#Image sets directory
if flag == 'day':
    train_set = ['set00', 'set01', 'set02']
    test_set = ['set06', 'set07', 'set08']
elif flag == 'night':
    train_set = ['set03', 'set04', 'set05']
    test_set = ['set09', 'set10', 'set11']

kaist_label_path = '/home/dlsj/Documents/KAIST_Dataset/annotations/'


#transformed lables to save path
kaist_label_tosave_path = '/home/dlsj/Documents/PyTorch-YOLOv3/data/kaist/labels/'
kaist_list_tosave_path = '/home/dlsj/Documents/PyTorch-YOLOv3/data/kaist/'

# index = 0
# cvfont = cv2.FONT_HERSHEY_SIMPLEX
'''
registering labeling to number
0 = person, 1 = people, 2 = cyclist
'''
#kaist_names = open('kaist.names','r')
kaist_names = open('kaist_person.names','r') #only person
kaist_names_contents = kaist_names.readlines()   
kaist_names_dic_key = []
for class_name in kaist_names_contents:
    kaist_names_dic_key.append(class_name.rstrip())
values = range(len(kaist_names_dic_key))
kaist_names_num = dict(zip(kaist_names_dic_key,values))


# kaist_images_folder = os.listdir(kaist_img_path) 

kaist_labels_folder = os.listdir(kaist_label_path)

# print(kaist_names_num)
# print(kaist_labels)

for phase in ['train','test']: #train_set, test_set
    print(phase)
    if phase=='train':
        allset = train_set
        f = open('{2}{1}_{0}.txt'.format(flag, phase, kaist_list_tosave_path),'w') #create list of images for training
    elif phase == 'test':
        allset = test_set 
        f = open('{2}val_{0}.txt'.format(flag, phase, kaist_list_tosave_path),'w') #create list of images for training
    kaist_img_path_phase = kaist_img_path + phase + '/' 
    for set_ in allset: #set0x
        kaist_img_path_set = kaist_img_path_phase + set_
        print(kaist_img_path_set) 
        
        #counter for each scenario
        set_counter = 0

        all_V = os.listdir(kaist_img_path_set)
        all_V.sort()

        for V00_ in all_V:
            #all train images but 20% from test 
            if set_ == 'set06' and set_counter == 901: #all = 1161, person only = 901
                print('{0} has reached {1}'.format(set_,set_counter))
                time.sleep(1)
                break
            elif set_ == 'set07' and set_counter == 514:
                print('{0} has reached {1}'.format(set_,set_counter))
                time.sleep(1)
                break
            elif set_ == 'set08' and set_counter == 942:
                print('{0} has reached {1}'.format(set_,set_counter))
                time.sleep(1)
                break
            elif set_ == 'set09' and set_counter == 545:
                print('{0} has reached {1}'.format(set_,set_counter))
                time.sleep(1)
                break
            elif set_ == 'set10' and set_counter == 561:
                print('{0} has reached {1}'.format(set_,set_counter))
                time.sleep(1)
                break
            elif set_ == 'set11' and set_counter == 324:
                print('{0} has reached {1}'.format(set_,set_counter))
                time.sleep(1)
                break

            # ex kaist_img_path = /home/dlsj/Documents/KAIST_Dataset/train/set02/
            # kaist_images = os.listdir(kaist_img_path_set + '/' + str(V00_) + '/visible/') #rgb images
            kaist_images = os.listdir(kaist_img_path_set + '/' + str(V00_) + '/lwir/') #infrared images
            kaist_labels = os.listdir(kaist_label_path + set_ + '/' + str(V00_) + '/')
            kaist_labels.sort()
            kaist_images.sort()
            print(V00_)
            #list of images
            for indexi, img in enumerate(kaist_images): #img is the name file, indexi is the iter
                #all train images but 20% from test 
                if set_ == 'set06' and set_counter == 901:
                    print('{0} has reached {1}'.format(set_,set_counter))
                    time.sleep(1)
                    break
                elif set_ == 'set07' and set_counter == 514:
                    print('{0} has reached {1}'.format(set_,set_counter))
                    time.sleep(1)
                    break
                elif set_ == 'set08' and set_counter == 942:
                    print('{0} has reached {1}'.format(set_,set_counter))
                    time.sleep(1)
                    break
                elif set_ == 'set09' and set_counter == 545:
                    print('{0} has reached {1}'.format(set_,set_counter))
                    time.sleep(1)
                    break
                elif set_ == 'set10' and set_counter == 561:
                    print('{0} has reached {1}'.format(set_,set_counter))
                    time.sleep(1)
                    break
                elif set_ == 'set11' and set_counter == 324:
                    print('{0} has reached {1}'.format(set_,set_counter))
                    time.sleep(1)
                    break

                filename = str(img).split('.png')
                # print(filename)
                filename = filename[0] #take the first name
                # print(filename)

                #kaist_img_totest_path = kaist_img_path_set + '/' + V00_ + '/visible/' + img #rgb
                kaist_img_totest_path = kaist_img_path_set + '/' + V00_ + '/lwir/' + img #infrared camera
                kaist_label_totest_path = kaist_label_path + set_ + '/' + V00_  + '/' + filename + '.txt' 
                # print(kaist_img_totest_path, kaist_label_totest_path)

                kaist_label_totest = open(kaist_label_totest_path,'r')
                
                label_contents = kaist_label_totest.readlines()
                # print(label_contents)
                #ex: /train/set02/V000/visible/
                # save_path = kaist_label_tosave_path + phase + '/' + set_ + '/' + V00_ + '/visible/'  #make a folder of V00/visible/
                save_path = kaist_label_tosave_path + phase + '/' + set_ + '/' + V00_ + '/lwir/'  #make a folder of V00/lwir/
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                
                for line in label_contents:
                    # print(len(line))
                    if (line=='% bbGt version=3\n'):
                    	continue #% bbGt version =3\n skip this part the first line
                    if line != '': #check if there must be an object

                        kaist_img_totest = cv2.imread(kaist_img_totest_path) #infrared
                        # print(kaist_img_totest,type(kaist_img_totest))
                        img_height, img_width = kaist_img_totest.shape[0],kaist_img_totest.shape[1]
                        # print(img_height, img_width)
                        
                        kaist_label_tosave = save_path + filename + '.txt'

                        data = line.split(' ')
                        
                        x=y=w=h=0
                        if(len(data) == 12):
                            # print(real_label)
                            # exit()
                            class_str = data[0] #class label
                            if class_str == 'person?' or class_str == 'person': 
                                class_str = 'person'
                            else: #uncomment this for excluding people and cyclist
                                continue #go to next line
                            #create a txt file for annotation
                            if os.path.exists(kaist_label_tosave):
                                real_label = open(kaist_label_tosave,'a') #append if it exists
                            else:
                                real_label = open(kaist_label_tosave, 'w') #make a new file if it doesn't exists
                                set_counter+=1 #add 1 more counter for 1 scenario
                                #save the image set if there is an object
                                print("writing to {2}{1}_{0}.txt".format(flag, phase, kaist_list_tosave_path))
                                f.write(kaist_img_totest_path + '\n')

                            # for kaist calls is a string
                            # trans this to number by using kaist.names
                            #(x,y) center (w,h) size
                            x1 = float(data[1]) 
                            y1 = float(data[2])
                            x2 = float(data[3]) #object width
                            y2 = float(data[4]) #object height

                            bbox_center_x = float( (x1 + (x2 / 2.0)) / img_width)  #anchor x
                            bbox_center_y = float( (y1 + (y2 / 2.0)) / img_height) #anchor y
                            bbox_width = float( x2 / img_width)
                            bbox_height = float( y2 / img_height)
                            # print(bbox_center_x, bbox_center_y, bbox_width, bbox_height)

                            #print(kaist_names_contents[class_num])
                
                            line_to_write = str(kaist_names_num[class_str]) + ' ' + str(bbox_center_x)+ ' ' + str(bbox_center_y)+ ' ' + str(bbox_width)+ ' ' + str(bbox_height) +'\n' #1 line of all texts
                            print('Writing {0} to {1}'.format(line_to_write,str(real_label)))
                            real_label.write(line_to_write) #write to the new file
                            sys.stdout.write(str(int((indexi/len(kaist_images))*100))+'% '+'*******************->' "\r" )
                            sys.stdout.flush()
                            real_label.close()
                

f.close()
kaist_names.close()
print("Labels transform finished!")