from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import linalg as LA
import math

dataset_im = "C:/Users/akharche/Desktop/VKR/IndianMovie/datasetDir.csv"
dataset_kinect = "C:/Users/akharche/Desktop/VKR/Kinect/dataDirKinect.csv"

train_data_im_tf2192 = "C:/Users/akharche/Desktop/VKR/IndianMovie/TF2192/IM_AgeGenderTF2192.csv"
train_data_im_tf2new = "C:/Users/akharche/Desktop/VKR/IndianMovie/TF2new/IM_AgeGenderTF2new.csv"

train_data_im_vgg16 = "C:/Users/akharche/Desktop/VKR/IndianMovie/vgg16/resGender.csv"
train_data_im_gendernet = "C:/Users/akharche/Desktop/VKR/IndianMovie/AgeGenderNet/IMAllImgCaffe.csv"


train_data_ijba_tf2192 = "C:/Users/akharche/Desktop/VKR/IJBA/TF2192/IJBA_AgeGenderTF2192.csv"

train_data_ijba_tf2new = "C:/Users/akharche/Desktop/VKR/IJBA/TF2new/IJBA_AgeGenderTF2new.csv"
train_data_ijba_vgg16 = "C:/Users/akharche/Desktop/VKR/IJBA/vgg16/IMDBIJBAllImgCaffe.csv"
train_data_ijba_gendernet = "C:/Users/akharche/Desktop/VKR/IJBA/AgeGenderNet/ijbAllImgCaffe.csv"

train_data_audience_tf2192 = "C:/Users/akharche/Desktop/VKR/TrainData/TrainDataTF2192.csv"
train_data_audience_tf2new = "C:/Users/akharche/Desktop/VKR/TrainData/TrainDataTF2new.csv"
dataset_audience = "C:/Users/akharche/Desktop/VKR/TrainData/dirAud_new.csv"

dataset_wiki = "C:/Users/akharche/Desktop/VKR/TrainData/imdbwiki.csv"
train_data_wikitf2192 = "C:/Users/akharche/Desktop/VKR/TrainData/imdbwiki_tf2192.csv"
train_data_wikitf2new = "C:/Users/akharche/Desktop/VKR/TrainData/imdbwiki_tf2new.csv"
train_data_wikilevi = "C:/Users/akharche/Desktop/VKR/TrainData/imdbwiki_genagenet1.csv"

dataset_emotiw =  "C:/Users/akharche/Desktop/VKR/Emotiw/datasetDirAll.csv"


ijba_data = "C:/Users/akharche/Desktop/VKR/IJBA/dir_new.csv"

age_list = [[0,2], [4,6], [8,12], [15,20], [25,32], [38,43], [48,53], [60,100]]

gender_list = ['male', 'female']

def arithmetic_average(preds):
    resulted_preds = np.sum(preds, axis=0) / len(preds)
    return resulted_preds

def generate_feature_map(dataset_path, index_value = 1):
    feature_map = {}

    dataset_frames = open(dataset_path).readlines()
    for frame in dataset_frames[0:]:
        try:
            frame = frame.split('\n')
            data = frame[0].split(',')
            current_frame_path = data[0]
            true_feature = data[index_value]
            feature_map[str(current_frame_path)] = true_feature
        except:
            pass
    return feature_map

def build_decision_templates(train_data_path, class_start, class_end):
    dt = []
    delim = ','
    train_predictions = open(train_data_path).readlines()
    dt_preds = []
    for tr_pred in train_predictions[0:]:
        parsed_line = tr_pred.split(delim)
        preds = [float(parsed_line[i]) for i in range(class_start, class_end)]
        dt_preds.append(preds)
    dt.append(arithmetic_average(dt_preds))
    return dt

def build_decision_templates_gender(train_data_path, dataset_path = ijba_data, class_start=2):
    gender_map = generate_feature_map(dataset_path)
    dt_male = []
    dt_female = []
    delim = ','
    train_predictions = open(train_data_path).readlines()
    dt = []
    for tr_pred in train_predictions[0:]:
        parsed_line = tr_pred.split(delim)
        current_img = parsed_line[0]
        preds = []
        preds.append(float(parsed_line[class_start]))
        preds.append(1 - float(parsed_line[class_start]))

        if (gender_map.get(current_img).lower() == 'male'):
            dt_male.append(preds)
        else:
            dt_female.append(preds)

    dt.append(arithmetic_average(dt_male))
    dt.append(arithmetic_average(dt_female))

    return dt

def change_dir(dataset_dir, dataset_dir_new):
    dirs = open(dataset_dir).readlines()
    out_file = open(dataset_dir_new, 'w')
    age_dict = {'(0, 2)': 1, '(4, 6)': 5, '(8, 12)': 10, '(15, 20)': 18, '(25, 32)': 28, '(38, 43)': 40, '(48, 53)': 50, '(60, 100)': 80}
    delimiter = ','

    for dir in dirs[0:]:
        try:
            line = dir.strip().split(delimiter,2)
            age_str = line[2]
            age_norm = age_dict.get(age_str)
            out_file.write(line[0]+delimiter+str(age_norm)+delimiter+'\n')
        except:
            pass
    out_file.close()

def norm_vect(age_preds, classes):
    norm_ages = []
    for i in range(0,len(classes)):
        norm_ages.append(age_preds[classes[i]])
    norm = LA.norm(norm_ages)
    return norm_ages/norm


def build_decision_templates_age(train_data_path, dataset_path=dataset_audience, class_start=5):
    #audience dataset as train: average age
    average_age_arr = [1,5,10,18,28,40,50,80]
    age_map = generate_feature_map(dataset_path, 1)
    dt_ages = []

    delim = ','
    train_predictions = open(train_data_path).readlines()
    dt = []
    for i in range(0, len(average_age_arr)):
        dt_ages = []
        for tr_pred in train_predictions[0:]:
            parsed_line = tr_pred.split(delim)
            current_img = parsed_line[0]
            a = age_map.get(current_img)
            b = str(average_age_arr[i])
            if(age_map.get(current_img) == str(average_age_arr[i])):
                age_preds = [float(i) for i in parsed_line[class_start:100+class_start]]
                ages_norm = norm_vect(age_preds, average_age_arr)
                dt_ages.append(ages_norm)
        dt.append(arithmetic_average(dt_ages))
    return dt

def build_decision_templates_fullage(train_data_path, dataset_path = dataset_wiki, class_start=4):

    age_map = generate_feature_map(dataset_path, 1)
    dt_ages = []

    ages = [i for i in range(1,101)]

    delim = ','
    train_predictions = open(train_data_path).readlines()
    dt = []
    for i, age in enumerate(ages):
        dt_ages = []
        for tr_pred in train_predictions[0:]:
            parsed_line = tr_pred.split(delim)
            current_img = parsed_line[0]
            if(age_map.get(current_img) == str(age)):
                age_preds = [float(i) for i in parsed_line[class_start:100+class_start]]
                dt_ages.append(age_preds)
        dt.append(arithmetic_average(dt_ages))
    return dt

def calculate_proximity(dt, predictions, num_of_classes=100):
    prox_classes = []
    for i in range(0,num_of_classes):
        class_preds = predictions
        class_dt = dt[i]
        norm_vect = np.power((1+LA.norm(np.subtract(class_dt, class_preds))), -1)
        #norm_vect = np.power((1 + np.sum(abs((np.subtract(class_dt, class_preds))), axis=0)), -1)
        prox_classes.append(norm_vect)
    norm_prox_classes = prox_classes/sum(prox_classes)
    return norm_prox_classes


def compute_belief_degrees(proximities, num_of_classes=2):
    belief_degrees = []
    current_classifier_prox = proximities
    for j in range(0, num_of_classes):
        class_mult = [(1-current_classifier_prox[k]) for k in range(0, num_of_classes) if k != j]
        num = (current_classifier_prox[j] * np.prod(class_mult))
        denom = (1 - current_classifier_prox[j])*(1-np.prod(class_mult))
        cl_ev = num / denom
        belief_degrees.append(cl_ev)
        print(np.sum(belief_degrees))
    return belief_degrees

def compute_b(proximities, num_of_classes=100):
    belief_degrees = []
    for j in range(0, num_of_classes):
        class_mult = [(1-proximities[k]) for k in range(0, num_of_classes) if k != j]
        #num = (proximities[j] * np.prod(class_mult))
        #denom = 1 - proximities[j]*(1-np.prod(class_mult))
        #cl_ev = (num / denom)
        num = np.log(proximities[j]) + np.sum(np.log(class_mult))
        denom = np.log(1-proximities[j]*(1-np.prod(class_mult)))
        cl_ev = num-denom
        belief_degrees.append(cl_ev)
    return belief_degrees

def final_decision(log_belief_degrees):
    #belief_degrees = np.log(np.asarray(belief_degrees))
   # belief_degrees = np.exp(np.log(np.asarray(belief_degrees)))
    #m = np.prod(belief_degrees, axis=0, dtype=np.float32)
    log_m = np.sum(log_belief_degrees, axis=0)
    #m = np.exp(log_m)
    m=log_m
    index = m.argsort()[::-1][:1]
    return index[0]

def build_dt_age(classes):
    dt = np.eye(classes)
    return dt


def aggregate_demster_shafer(preds_path, out_preds_path, dataset_path):
    #dt = build_decision_templates(dataset_path, 4,104)
    dt = build_decision_templates_gender(dataset_path,class_start=2)

    dtfile = open('C:/Users/akharche/Desktop/dt_gend.csv', 'w')
    for dt_i in range(0, len(dt)):
            for i in range(0, len(dt[dt_i])):
                dtfile.write(str(dt[dt_i][i])+',')
            dtfile.write('\n')
    dtfile.close()

    #dt = [[1,0],[0,1]]
    preds_results = open(preds_path).readlines()

    beliefs = []

    delimiter = ','

    #video_dir = preds_results[0].split('\\\\')[6]
    video_dir = preds_results[0].split('/')[4]
    output_file = open(out_preds_path, 'w')
    line_counter = 0

    for pred in preds_results[0:]:
        #try:

            #frame_dir = pred.strip().split('\\\\')
            frame_dir = pred.strip().split('/')
            current_video_dir = frame_dir[4]
            line_counter += 1

            if (current_video_dir != video_dir) | (line_counter == len(preds_results)):
                output_file.write(line[0] + delimiter)

                gender = final_decision(beliefs)
                output_file.write(gender_list[gender]+delimiter)

                #beliefs = compute_belief_degrees(all_proximities, 100)
                #age = final_decision(beliefs)
                #output_file.write(str(age))
                output_file.write('\n')
                all_proximities = []
                beliefs = []

            gender_preds = []
            line = pred.split(',')
            gender_preds.append(float(line[2]))
            gender_preds.append(1-float(line[2]))
            gender_proximities = calculate_proximity(dt, gender_preds, 2)
            b = compute_b(gender_proximities,2)
            beliefs.append(b)

            #age_preds = [float(i) for i in line[4:104]]

            '''
            age_proximities = calculate_proximity(dt, age_preds, 100)
            b = compute_b(age_proximities)
            #all_proximities.append(age_proximities)
            beliefs.append(b)
            '''

            video_dir = current_video_dir
        #except:
           # pass
    output_file.close()

def aggregate_demster_shafer_age(preds_path, out_preds_path, dataset_path = train_data_ijba_tf2192):
    #dt = build_decision_templates(dataset_path, 4,104)
    #dt = build_decision_templates_gender(dataset_path,class_start=1)
    dt = build_dt_age(100)
    preds_results = open(preds_path).readlines()

    beliefs = []

    delimiter = ','

    #video_dir = preds_results[0].split('\\\\')[5]
    video_dir = preds_results[0].split('/')[4]
    output_file = open(out_preds_path, 'w')
    line_counter = 0

    for pred in preds_results[0:]:
        #try:

            #frame_dir = pred.strip().split('\\\\')
            frame_dir = pred.strip().split('/')
            current_video_dir = frame_dir[4]
            line_counter += 1

            if (current_video_dir != video_dir) | (line_counter == len(preds_results)):
                output_file.write(line[0] + delimiter)

                #beliefs = compute_belief_degrees(all_proximities, 100)
                age = final_decision(beliefs)
                output_file.write(str(age))
                output_file.write('\n')
                all_proximities = []
                beliefs = []

            line = pred.split(',')

            age_preds = [float(i) for i in line[4:104]]

            age_proximities = calculate_proximity(dt, age_preds, 100)
            b = compute_b(age_proximities, 100)
            beliefs.append(b)
            video_dir = current_video_dir
        #except:
           # pass
    output_file.close()

def aggregate_demster_shafer_age_dt(preds_path, out_preds_path, dataset_path = train_data_ijba_tf2192):
    dt = build_decision_templates_age(dataset_path, dataset_audience)

    average_age_arr = [1, 5, 10, 18, 35, 40, 50, 80]

    preds_results = open(preds_path).readlines()

    beliefs = []

    delimiter = ','

    #video_dir = preds_results[0].split('\\\\')[5]

    video_dir = preds_results[0].split('/')[4]
    output_file = open(out_preds_path, 'w')
    line_counter = 0

    for pred in preds_results[0:]:
        #try:

            frame_dir = pred.strip().split('/')
            current_video_dir = frame_dir[4]
            line_counter += 1

            if (current_video_dir != video_dir) | (line_counter == len(preds_results)):
                output_file.write(line[0] + delimiter)

                age = final_decision(beliefs)
                age_res = average_age_arr[age]
                output_file.write(str(age_res))
                output_file.write('\n')

                beliefs = []

            line = pred.split(',')

            age_preds = [float(i) for i in line[4:104]]
            ages_norm = norm_vect(age_preds, average_age_arr)

            age_proximities = calculate_proximity(dt, ages_norm, 8)
            b = compute_b(age_proximities, 8)
            beliefs.append(b)
            video_dir = current_video_dir
        #except:
           # pass
    output_file.close()

def aggregate_demster_shafer_age_dt_fullage(preds_path, out_preds_path, train_dataset_path):
    #dt = build_decision_templates_fullage(train_dataset_path)

    dt = []
    dtfile = open('C:/Users/akharche/Desktop/dt.csv').readlines()
    for line in dtfile[0:]:
        dt_i = line.split('\n')[0].split(',')
        tmp = [float(i) for i in dt_i[0:100]]
        dt.append(tmp)

    preds_results = open(preds_path).readlines()
    beliefs = []
    delimiter = ','

    #video_dir = preds_results[0].split('\\\\')[5]
    video_dir = preds_results[0].split('/')[4]
    output_file = open(out_preds_path, 'w')
    line_counter = 0

    for pred in preds_results[0:]:
        #try:
            #frame_dir = pred.strip().split('\\\\')
            frame_dir = pred.strip().split('/')
            current_video_dir = frame_dir[4]
            line_counter += 1

            if (current_video_dir != video_dir) | (line_counter == len(preds_results)):
                output_file.write(line[0] + delimiter)

                #beliefs = compute_belief_degrees(all_proximities, 100)
                age = final_decision(beliefs)
                output_file.write(str(age))
                output_file.write('\n')
                beliefs = []

            line = pred.split(',')

            age_preds = [float(i) for i in line[4:104]]
            age_proximities = calculate_proximity(dt, age_preds, 100)
            b = compute_b(age_proximities, 100)
            beliefs.append(b)
            video_dir = current_video_dir
        #except:
           # pass
    output_file.close()

if __name__ == '__main__':
    print("Demster Shafer aggregation:\n")

    #aggregate_demster_shafer_age_dt_fullage("C:/Users/akharche/Desktop/VKR/Emotiw/emotiw_tf2new.csv", "C:/Users/akharche/Desktop/VKR/Emotiw/DS_tf2new_AGE_DTWIKI.csv", train_data_wikitf2new)
    aggregate_demster_shafer("C:/Users/akharche/Desktop/VKR/Emotiw/emotiw_tf2new.csv", "C:/Users/akharche/Desktop/VKR/Emotiw/gend_emotiw_tf2new.csv", train_data_ijba_tf2new)

