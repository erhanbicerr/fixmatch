import os 
import re
import numpy as np
#angry/10002-angry_F-EA-03/0.png
# fold1_im_list = os.listdir("ML_DATA/CAFE_5emotions_augmented_format/fold1_images")
# fold2_im_list = os.listdir("ML_DATA/CAFE_5emotions_augmented_format/fold2_images")
# fold3_im_list = os.listdir("ML_DATA/CAFE_5emotions_augmented_format/fold3_images")

def original_image(path_fiz):
    #path_fiz='Dosya_sirasi_fizyo/LABEL_name_U07.txt'
    original_label_x=[]
    count=0
    with open(path_fiz) as f_m:
        lines_m= f_m.readlines()
        #print(lines_m[1])
        for line in lines_m:
            x=re.split('/', line.strip())
            conc_data=re.split('-', x[1])
            original_label_x.append(conc_data[0])
            print(conc_data[0])

    print(np.shape(original_label_x))
    return original_label_x 

def fold_image(path_fiz):
    #path_fiz='Dosya_sirasi_fizyo/LABEL_name_U07.txt'
    original_label_x=[]
    count=0
    fold_im = os.listdir(path_fiz)    

    #print(lines_m[1])
    for line in fold_im:
        x=re.split('_', line.strip())
        conc_data=re.split('SN', x[1])
        original_label_x.append(conc_data[1])
        print(conc_data[1])

    print(np.shape(original_label_x))
    return original_label_x 

def original_label(path_fiz):
    #path_fiz='Dosya_sirasi_fizyo/LABEL_name_U07.txt'
    original_label_x=[]
    count=0
    with open(path_fiz) as f_m:
        lines_m= f_m.readlines()
        #print(lines_m[1])
        for line in lines_m:
            label = np.zeros(5)
            line_split = re.split(' ', line.strip())
            label = np.array(line_split).astype(int)
            label_sayi = np.argmax(label)
            if (label_sayi>=3):
                label_sayi-=1
            original_label_x.append(label_sayi)
            print(label_sayi)

    print(np.shape(original_label_x))
    print(np.unique(np.array(original_label_x)))
    return original_label_x 

def label_matcher(fold_data,org_data, org_label):
    #org_data: drive (emotion)
    #fold_data: AU
    new_label_list = []
    print(f"Count of fold sample:{len(fold_data)}")
    for sn_idx in range(np.shape(fold_data)[0]):
        #print(sn_idx)
        #print(len(fold_data[sn_idx]))
        #print(len(org_data[0]))
        index_found=np.where(np.array(org_data)==fold_data[sn_idx])[0][0] # get a random one because all labels are same for SN
        #print(index_found)
        new_label_list.append(org_label[index_found])
    #print(new_label_list)
    #print(len(new_label_list))
    #print(type(new_label_list))
    print(f"Count of matched labels:{len(new_label_list)}")
    new_label_list = np.array(new_label_list)
    return new_label_list
def fold_image(path_fiz):
    #path_fiz='Dosya_sirasi_fizyo/LABEL_name_U07.txt'
    original_subjectnum=[]
    count=0
    fold_im = os.listdir(path_fiz)    

    #print(lines_m[1])
    for line in fold_im:
        x=re.split('_', line.strip())
        conc_data=re.split('SN', x[1])
        subject_num = conc_data[1]
        original_subjectnum.append(subject_num)
        print(subject_num)

    print(np.shape(original_subjectnum))
    return original_subjectnum 

im_list_org = original_image("train_imgs.txt") # 9087...
im_label_org = original_label("train_labels.txt")

#print(im_label_org)
#print(len(im_list_org)) # org img ismi
#print(len(im_label_org)) # org img labellari

#print(im_list_org)
fold1_im_list = fold_image("ML_DATA/CAFE_5emotions_augmented_format/fold1_images")
fold2_im_list = fold_image("ML_DATA/CAFE_5emotions_augmented_format/fold2_images")
fold3_im_list = fold_image("ML_DATA/CAFE_5emotions_augmented_format/fold3_images")
#print(len(fold3_im_list))

new_label1 = label_matcher(fold1_im_list,im_list_org,im_label_org)
np.save("part_1_label_array_emotion.npy",new_label1)

new_label2 = label_matcher(fold2_im_list,im_list_org,im_label_org)
np.save("part_2_label_array_emotion.npy",new_label2)

new_label3 = label_matcher(fold3_im_list,im_list_org,im_label_org)
np.save("part_3_label_array_emotion.npy",new_label3)
