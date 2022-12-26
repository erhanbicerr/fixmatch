import os 
import re
import numpy as np
import pandas as pd
emotion_map = {"angry":0,
"happy":1,
"sad":2,
"sadopen":2,
"surprise":3}

def get_sn_label_org(path):
    """
    
    get subject number and label of original emotion data
    """
    subject_nums=[]
    labels = []

    with open(path) as text_file:
        lines = text_file.readlines()

        for line in lines:
            im_splitted=re.split('/', line.strip())
            im_splitted=re.split('-', im_splitted[1])
            emotion = im_splitted[1].split("_")[0]
            sn = im_splitted[0] # subject/capture number
            if (emotion=="neutral"):
                print(emotion)
                print(sn,"skipped",sep=" ")
                continue
            #print(sn)
            subject_nums.append(sn)
            emotion_label = emotion_map[emotion]
            labels.append(emotion_label)

    print("SN:",np.shape(subject_nums))
    print("Labels:",np.shape(labels))
    return subject_nums,labels

def get_path_foldimages(path_fiz):
    #path_fiz='Dosya_sirasi_fizyo/LABEL_name_U07.txt'
    original_label_x=[]
    count=0
    fold_im_list = os.listdir(path_fiz)    
    return fold_im_list

def label_matcher_fromDF(fold_im_path,org_data_df):
    #org_data: drive (emotion)
    #fold_im_path: AU
    fold_DF = pd.DataFrame()
    count = 0
    new_label_list = []
    print(f"Count of fold sample:{len(fold_im_path)}")


    for im_fname in fold_im_path:
        print(im_fname)
        im_fname_splitted=re.split('_', im_fname.strip())
        im_fname_splitted=re.split('SN', im_fname_splitted[1])
        sn_fold = im_fname_splitted[1]

        new_label = org_data_df[org_data_df["im_name"]==sn_fold]["label"].unique()[0]
        print(new_label)
        fold_DF.loc[count,"im_path"] = im_fname
        fold_DF.loc[count,"label"] = new_label
        count+=1

    print(f"Count of matched labels:{len(fold_DF)}")
    print(f"Count of emotions:{fold_DF.label.value_counts()}")

    return fold_DF


im_list_org, im_label_org  = get_sn_label_org("train_imgs.txt") # 2344 images
print(np.unique(im_label_org,return_counts=True))
org_dataframe = pd.DataFrame()
org_dataframe["im_name"] = im_list_org
org_dataframe["label"] = im_label_org
print(org_dataframe)


fold1_im_list = get_path_foldimages("ML_DATA/CAFE_5emotions_augmented_format/fold1_images")
fold_DF = label_matcher_fromDF(fold1_im_list,org_dataframe)
fold_DF.to_csv("part_1_label_array_emotion.csv",index=False)
print(fold_DF.head(10))
print(fold_DF.tail(10))

fold2_im_list = get_path_foldimages("ML_DATA/CAFE_5emotions_augmented_format/fold2_images")
fold_DF = label_matcher_fromDF(fold2_im_list,org_dataframe)
fold_DF.to_csv("part_2_label_array_emotion.csv",index=False)
print(fold_DF.head(10))
print(fold_DF.tail(10))

fold3_im_list = get_path_foldimages("ML_DATA/CAFE_5emotions_augmented_format/fold3_images")
fold_DF = label_matcher_fromDF(fold3_im_list,org_dataframe)
fold_DF.to_csv("part_3_label_array_emotion.csv",index=False)
print(fold_DF.head(10))
print(fold_DF.tail(10))

