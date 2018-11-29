from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import math
import time

gender_list = ['male', 'female']
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
age_list_average = ['1', '5', '10', '18', '29', '41', '51', '80']


def is_male(gender_pred):
    return (gender_pred)>=0.6

def get_gender(is_male):
    return 'male' if is_male else 'female'

def ident_gender(gender_pred):
    if is_male(gender_pred):
        return 'male'
    else:
         return  'female'

def get_age(age_preds):
    index = age_preds.argsort()[::-1][:1]
    return str(index[0])

def aggregate_expected_value(age_preds):
    min_age = 1
    age_preds = np.array(age_preds)
    indices = age_preds.argsort()[::-1][:2]

    norm_preds = age_preds[indices] / np.sum(age_preds[indices])

    res_age = min_age
    for age, probab in zip(indices, norm_preds):
        res_age += age * probab
    return res_age

def aggregate_gender_simple_vote(preds):

    votes_to_classes = [0, 0]
    male_pos = 0
    female_pos = 1
    for p in preds[0:]:
        if is_male(p[0]):
            votes_to_classes[male_pos] += 1
        else:
            votes_to_classes[female_pos] += 1

    result_index = (np.array(votes_to_classes)).argsort()[::-1][:1]

    return gender_list[result_index[0]]

def aggregate_age_simple_vote(preds, num_of_classes = 8):

    votes_to_classes = np.zeros((1, num_of_classes))
    for p in preds[0:]:
        max_index = (np.array(p)).argsort()[::-1][:1]

        votes_to_classes[0][max_index] += 1

    result_index = (votes_to_classes[0]).argsort()[::-1][:1]

    return result_index[0]


def aggregate_arithmetic_average(preds):

    resulted_preds = np.sum(preds, axis=0) / len(preds)
    return resulted_preds

def aggregate_geometric_average(preds):

    resulted_preds = np.exp(np.sum(np.log(preds), axis=0) / len(preds))
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

#TODO
def read_video_dir(dataset_dir):

    if dataset_dir == 'kinect':
        video_index = 5
        gender_index = 1
        age_index = 4
        delim = '\\\\'

    return video_index, gender_index, age_index, delim

# GENDER_NET AGE_NET
def get_from_list(preds, list):
    ind = preds.argsort()[::-1][:1]
    return list[ind[0]]

def aggregate_gender_simple_vote_arr(preds):
    votes_to_classes = [0, 0]

    for p in preds[0:]:
        tmp = np.array(p)
        ind = tmp.argsort()[::-1][:1]
        votes_to_classes[ind[0]] += 1

    result_index = (np.array(votes_to_classes)).argsort()[::-1][:1]
    return gender_list[result_index[0]]

def aggregate_expected_value_agenet(age_preds):
    min_age = 1
    age_preds = np.array(age_preds)
    indices = age_preds.argsort()[::-1][:2]

    norm_preds = age_preds[indices] / np.sum(age_preds[indices])
    middle_age = [1, 5, 10, 18, 29, 41, 51, 80]

    res_age = min_age
    for age, probab in zip(indices, norm_preds):
        res_age += middle_age[age] * probab
    return res_age

# GENDER_NET AGE_NET
def aggregate_probabs_agegendernet(preds_path, aggregated_preds_path):
    preds_results = open(preds_path).readlines()

    gender_preds_to_aggreg = []
    age_preds_to_aggreg = []
    age_agreg_expectedvalue = []
    tmp = []
    delimiter = ','

    video_dir = preds_results[0].strip().split('\\\\')[5]

    output_file = open(aggregated_preds_path, 'w')

    line_counter = 0

    for pred in preds_results[0:]:
        # try:
        frame_dir = pred.strip().split('\\\\')
        current_video_dir = frame_dir[5]
        line_counter += 1

        if (current_video_dir != video_dir) | (line_counter >= len(preds_results)):
            output_file.write(line[0] + ',')

            gender_sv = aggregate_gender_simple_vote_arr(gender_preds_to_aggreg)
            gender_am = get_from_list(aggregate_arithmetic_average(gender_preds_to_aggreg), gender_list)
            gender_gm = get_from_list(aggregate_geometric_average(gender_preds_to_aggreg), gender_list)
            gender_results = (gender_sv, gender_am, gender_gm)
            output_file.write(delimiter.join(gender_results) + delimiter)

            age_me = (aggregate_arithmetic_average(tmp))
            age_ind_sv = aggregate_age_simple_vote(age_preds_to_aggreg, 8)
            age_sv = age_list_average[age_ind_sv]
            age_am = get_from_list(aggregate_arithmetic_average(age_preds_to_aggreg), age_list_average)
            age_gm = get_from_list(aggregate_geometric_average(age_preds_to_aggreg), age_list_average)

            age_results = (str(int(age_me[0])), str(age_sv), str(age_am), str(age_gm))

            output_file.write(delimiter.join(age_results))
            output_file.write('\n')

            gender_preds_to_aggreg = []
            age_preds_to_aggreg = []
            age_agreg_expectedvalue = []
            tmp = []

        gender_preds = []
        line = pred.split(',')
        gender_preds.append(float(line[2]))
        gender_preds.append(float(line[3]))
        gender_preds_to_aggreg.append(gender_preds)

        age_preds = []
        for i in line[5:13]:
            age_preds.append(float(i))


        age_agreg_expectedvalue.append(aggregate_expected_value_agenet(age_preds))
        tmp.append(age_agreg_expectedvalue)
        age_preds_to_aggreg.append(age_preds)

        video_dir = current_video_dir
    # except:
    # pass

    output_file.close()


def aggregate_probabs(preds_path, aggregated_preds_path):

    preds_results = open(preds_path).readlines()

    gender_preds_to_aggreg = []
    age_preds_to_aggreg = []
    age_agreg_expectedvalue = []
    tmp = []
    delimiter = ','

    video_dir = preds_results[0].split('\\\\')[5]

    output_file = open(aggregated_preds_path, 'w')

    line_counter = 0

    for pred in preds_results[0:]:
        try:
            frame_dir = pred.split('\\\\')
            current_video_dir = frame_dir[5]
            line_counter += 1
            print(pred)


            if (current_video_dir != video_dir) | (line_counter >= len(preds_results)):
                output_file.write(line[0] + ',')

                gender_sv = aggregate_gender_simple_vote(gender_preds_to_aggreg)
                gender_am = ident_gender(aggregate_arithmetic_average(gender_preds_to_aggreg))
                gender_gm = ident_gender(aggregate_geometric_average(gender_preds_to_aggreg))
                gender_results = (gender_sv, gender_am, gender_gm)
                output_file.write(delimiter.join(gender_results) + delimiter)


                age_sv = aggregate_age_simple_vote(age_preds_to_aggreg, 100)
                age_am = get_age(aggregate_arithmetic_average(age_preds_to_aggreg))
                age_gm = get_age(aggregate_geometric_average(age_preds_to_aggreg))
                age_me = (aggregate_arithmetic_average(tmp))
                age_results = (str(int(age_me[0])), str(age_sv), str(age_am), str(age_gm))


                output_file.write(delimiter.join(age_results))
                output_file.write('\n')


                gender_preds_to_aggreg = []
                age_preds_to_aggreg = []
                age_agreg_expectedvalue = []
                tmp = []

            gender_preds = []
            line = pred.split(',')
            gender_preds.append(float(line[2]))
            gender_preds_to_aggreg.append(gender_preds)

            age_preds = []
            for i in line[3:104]:
                age_preds.append(float(i))

            age_agreg_expectedvalue.append(aggregate_expected_value(age_preds))
            tmp.append(age_agreg_expectedvalue)
            age_preds_to_aggreg.append(age_preds)

            video_dir = current_video_dir

        except:
            pass

    output_file.close()


def calculate_accuracy(dataset_path, predictions_path, out_data_file):

    gender_map = generate_feature_map(dataset_path)
    age_map = generate_feature_map(dataset_path, 2)

    year_today = 2014

    preds_results = open(predictions_path).readlines()
    delimiter = ','

    gender_err = 0
    age_err = []
    age_errfbf = []

    age_intervals_for_acccur = [0, 3, 5, 10, 15, 20]

    age_err_inter = np.zeros(len(age_intervals_for_acccur))

    for pred in preds_results[0:]:
        line = pred.split(',')
        current_frame = line[0]
        gender = line[1]
        age = float((line[3])[:5])

        age_preds = []
        for i in line[4:104]:
            age_preds.append(float(i))

        age_preds = np.expand_dims(age_preds, axis=0)
        agefbf = get_age(age_preds[0])


        if gender != gender_map.get(current_frame).lower():
            gender_err += 1

        for index, interval in enumerate(age_intervals_for_acccur):
            true_age = year_today - int(age_map.get(current_frame))
            if not ((true_age - interval) <= int(agefbf) <= (true_age + interval)):
                age_err_inter[index] += 1

        # MAE error for age
        true_age = year_today - int(age_map.get(current_frame))
        difference_in_age = (np.abs(true_age - age))
        age_err.append(difference_in_age)

        difference_in_agefbf = (np.abs(true_age - int(agefbf)))
        age_errfbf.append(difference_in_agefbf)

    output_file = open(out_data_file, 'w')

    mae = aggregate_arithmetic_average(age_err)
    maefbf = aggregate_arithmetic_average(age_errfbf)

    output_file.write(str(gender_err) + delimiter)
    output_file.write(str(mae) + delimiter)
    output_file.write(str(maefbf) + delimiter)
    output_file.write(str(age_err_inter))
    output_file.close()

#KINECT


def calculate_accuracy_aggreg(dataset_path, predictions_path, out_data, dataset_name = 'kinect'):

    gender_map = generate_feature_map(dataset_path)
    age_map = generate_feature_map(dataset_path, 2)

    year_today = 2014

    preds_results = open(predictions_path).readlines()
    delimiter = ','

    sv_gender_err = 0
    am_gender_err = 0
    gm_gender_err = 0

    age_intervals_for_acccur = [0, 3, 5, 10, 15, 20]

    me_age_err_inter = np.zeros(len(age_intervals_for_acccur))
    sv_age_err_inter = np.zeros(len(age_intervals_for_acccur))
    am_age_err_inter = np.zeros(len(age_intervals_for_acccur))
    gm_age_err_inter = np.zeros(len(age_intervals_for_acccur))

    #MAE

    me_age_err = []
    sv_age_err = []
    am_age_err = []
    gm_age_err = []


    for pred in preds_results[0:]:
        #try:
            line = pred.split(',')
            current_dir = line[0]

            gender_sv = line[1]
            gender_am = line[2]
            gender_gm = line[3]

            age_me = int(line[4])
            age_sv = int(line[5])
            age_am = int(line[6])
            age_gm = int(line[7])

            if gender_sv != gender_map.get(current_dir).lower():
                sv_gender_err += 1
            if gender_am != gender_map.get(current_dir).lower():
                am_gender_err += 1
            if gender_gm != gender_map.get(current_dir).lower():
                gm_gender_err += 1


            for index, interval in enumerate(age_intervals_for_acccur):
                true_age = year_today - int(age_map.get(current_dir))
                if not ((true_age - interval) <= int(age_me) <= (true_age + interval)):
                     me_age_err_inter[index] += 1

                if not ((true_age - interval) <= int(age_sv) <= (true_age + interval)):
                    sv_age_err_inter[index] += 1
                if not ((true_age - interval) <= int(age_am) <= (true_age + interval)):
                    am_age_err_inter[index] += 1
                if not ((true_age - interval) <= int(age_gm) <= (true_age + interval)):
                    gm_age_err_inter[index] += 1


            # MAE error for age
            true_age = year_today - int(age_map.get(current_dir))

            difference_in_age_me = (np.abs(true_age - age_me))
            me_age_err.append(difference_in_age_me)

            difference_in_age_sv = (np.abs(true_age - age_sv))
            sv_age_err.append(difference_in_age_sv)

            difference_in_age_am = (np.abs(true_age - age_am))
            am_age_err.append(difference_in_age_am)

            difference_in_age_gm = (np.abs(true_age - age_gm))
            gm_age_err.append(difference_in_age_gm)
        #except:
            #pass

    mae_me = aggregate_arithmetic_average(me_age_err)
    mae_sv = aggregate_arithmetic_average(sv_age_err)
    mae_am = aggregate_arithmetic_average(am_age_err)
    mae_gm = aggregate_arithmetic_average(gm_age_err)

    output_file = open(out_data, 'w')
    err_gender_res = (str(sv_gender_err), str(am_gender_err), str(gm_gender_err))
    mae_age_res = (str(mae_me), str(mae_sv), str(mae_am), str(mae_gm))
    err_age_res = (str(me_age_err_inter), str(sv_age_err_inter), str(am_age_err_inter), str(gm_age_err_inter))
    output_file.write(delimiter.join(err_gender_res) + delimiter)
    output_file.write(delimiter.join(mae_age_res) + delimiter)
    output_file.write(delimiter.join(err_age_res))
    output_file.close()



# Indian Movie
'''
Age: Child (1 − 12 years), Young (13 − 30 years),
Middle (31 − 50 years), Old (Above 50 years)
'''
def calculate_accuracy_im(dataset_path, predictions_path, out_data):

    gender_map = generate_feature_map(dataset_path)
    age_map = generate_feature_map(dataset_path, 2)

    true_age_map = {'CHILD': int(6), 'YOUNG': int(22), 'MIDDLE': int(40), 'OLD': int(75)}

    preds_results = open(predictions_path).readlines()
    delimiter = ','

    age_intervals_for_acccur = [0, 3, 5, 10, 15, 20]

    age_err_inter = np.zeros(len(age_intervals_for_acccur))

    gender_err = 0
    age_err = []
    age_err_fbf = []
    tmp = 0
    for pred in preds_results[0:]:
        try:
            line = pred.split(',')
            current_frame = line[0]
            gender = line[1]

            age = float((line[3])[:5])

            if gender != gender_map.get(current_frame).lower():
                gender_err += 1

            age_preds = []
            for i in line[4:104]:
                age_preds.append(float(i))

            age_preds = np.expand_dims(age_preds, axis=0)
            agefbf = get_age(age_preds[0])

            age_interval = age_map.get(current_frame)
            true_age = true_age_map.get(age_interval)

            for index, interval in enumerate(age_intervals_for_acccur):
                if not ((true_age - interval) <= int(agefbf) <= (true_age + interval)):
                    age_err_inter[index] += 1

            # MAE error for age

            difference_in_age = (np.abs(true_age - age))
            age_err.append(difference_in_age)

            difference_in_agefbf = (np.abs(true_age - int(agefbf)))
            age_err_fbf.append(difference_in_agefbf)

            tmp += 1

        except:
            pass

    output_file = open(out_data, 'w')

    mae = aggregate_arithmetic_average(age_err)
    maefbf = aggregate_arithmetic_average(age_err_fbf)

    output_file.write(str(gender_err) + delimiter)
    output_file.write(str(mae) + delimiter)
    output_file.write(str(maefbf) + delimiter)
    output_file.write(str(age_err_inter))
    output_file.close()


def calculate_accuracy_aggreg_im(dataset_path, predictions_path, out_data):

    gender_map = generate_feature_map(dataset_path)
    age_map = generate_feature_map(dataset_path, 2)

    true_age_map = {'CHILD': [1,12], 'YOUNG': [13,30], 'MIDDLE': [31,50], 'OLD': [51,100]}

    #true_age_map = {'CHILD': int(12), 'YOUNG': int(30), 'MIDDLE': int(50), 'OLD': int(75)}

    preds_results = open(predictions_path).readlines()
    delimiter = ','

    sv_gender_err = 0
    am_gender_err = 0
    gm_gender_err = 0

    #MAE
    me_age_err = []
    sv_age_err = []
    am_age_err = []
    gm_age_err = []

    #age_intervals_for_acccur = [0, 3, 5, 10, 15, 20]

    #me_age_err_inter = np.zeros(len(age_intervals_for_acccur))
    #sv_age_err_inter = np.zeros(len(age_intervals_for_acccur))
    #am_age_err_inter = np.zeros(len(age_intervals_for_acccur))
    #gm_age_err_inter = np.zeros(len(age_intervals_for_acccur))

    me_age_err_inter = 0
    sv_age_err_inter = 0
    am_age_err_inter = 0
    gm_age_err_inter = 0

    for pred in preds_results[0:]:
        #try:
            line = pred.strip().split(',')
            current_dir = line[0]

            gender_sv = line[1]
            gender_am = line[2]
            gender_gm = line[3]

            age_me = int(line[4])
            age_sv = int(line[5])
            age_am = int(line[6])
            age_gm = int(line[7])

            if gender_sv != gender_map.get(current_dir).lower():
                sv_gender_err += 1
            if gender_am != gender_map.get(current_dir).lower():
                am_gender_err += 1
            if gender_gm != gender_map.get(current_dir).lower():
                gm_gender_err += 1

            age_interval = age_map.get(current_dir)
            age = true_age_map.get(age_interval)
            true_age_min = age[0]
            true_age_max = age[1]

            if not (true_age_min <= int(age_me) <= true_age_max):
                me_age_err_inter += 1
            if not (true_age_min <= int(age_sv) <= true_age_max):
                sv_age_err_inter += 1
            if not (true_age_min <= int(age_am) <= true_age_max):
                am_age_err_inter += 1
            if not (true_age_min <= int(age_gm) <= true_age_max):
                gm_age_err_inter += 1

            '''
            for index, interval in enumerate(age_intervals_for_acccur):
                if not ((true_age - interval) <= int(age_me) <= (true_age + interval)):
                     me_age_err_inter[index] += 1
                if not ((true_age - interval) <= int(age_sv) <= (true_age + interval)):
                    sv_age_err_inter[index] += 1
                if not ((true_age - interval) <= int(age_am) <= (true_age + interval)):
                    am_age_err_inter[index] += 1
                if not ((true_age - interval) <= int(age_gm) <= (true_age + interval)):
                    gm_age_err_inter[index] += 1

            '''

            # MAE error for age
            '''
            difference_in_age_me = (np.abs(true_age - age_me))
            print("true " + str(true_age))
            print(age_me)
            print(difference_in_age_me)
            me_age_err.append(difference_in_age_me)

            difference_in_age_sv = (np.abs(true_age - age_sv))
            sv_age_err.append(difference_in_age_sv)

            difference_in_age_am = (np.abs(true_age - age_am))
            am_age_err.append(difference_in_age_am)

            difference_in_age_gm = (np.abs(true_age - age_gm))
            gm_age_err.append(difference_in_age_gm)
        #except:
            #pass

    mae_me = aggregate_arithmetic_average(me_age_err)
    print(mae_me)
    mae_sv = aggregate_arithmetic_average(sv_age_err)
    mae_am = aggregate_arithmetic_average(am_age_err)
    mae_gm = aggregate_arithmetic_average(gm_age_err)
    '''

    output_file = open(out_data, 'w')
    err_gender_res = (str(sv_gender_err), str(am_gender_err), str(gm_gender_err))
    #mae_age_res = (str(mae_me), str(mae_sv), str(mae_am), str(mae_gm))
    err_age_res = (str(me_age_err_inter), str(sv_age_err_inter), str(am_age_err_inter), str(gm_age_err_inter))
    output_file.write(delimiter.join(err_gender_res) + delimiter)
    #output_file.write(delimiter.join(mae_age_res) + delimiter)
    output_file.write(delimiter.join(err_age_res))
    output_file.close()

# IJBA

def make_dir(data_set_path, new_dataset_path):

    frames = open(data_set_path).readlines()
    out_data = open(new_dataset_path, 'w')

    gender_dict = {'1': 'male', '0': 'female'}

    for frame in frames[0:]:
        gender = frame.split('\\\\')[5]
        out_data.write(frame.split('\n')[0] + ',')
        out_data.write(gender_dict.get(gender) + '\n')

    out_data.close()


def aggregate_probabs_ijba(preds_path, aggregated_preds_path):

    preds_results = open(preds_path).readlines()

    gender_preds_to_aggreg = []
    delimiter = ','

    video_dir = preds_results[0].split('\\\\')[6]

    output_file = open(aggregated_preds_path, 'w')

    line_counter = 0

    for pred in preds_results[0:]:
        #try:
            frame_dir = pred.split('\\\\')
            current_video_dir = frame_dir[6]
            line_counter += 1

            if (current_video_dir != video_dir) | (line_counter >= len(preds_results)):
                output_file.write(line[0] + ',')

                gender_sv = aggregate_gender_simple_vote(gender_preds_to_aggreg)
                gender_am = ident_gender(aggregate_arithmetic_average(gender_preds_to_aggreg))
                gender_gm = ident_gender(aggregate_geometric_average(gender_preds_to_aggreg))
                gender_results = (gender_sv, gender_am, gender_gm)
                output_file.write(delimiter.join(gender_results) + delimiter)
                output_file.write('\n')

                gender_preds_to_aggreg = []

            gender_preds = []
            line = pred.split(',')
            gender_preds.append(float(line[2]))
            gender_preds_to_aggreg.append(gender_preds)
            video_dir = current_video_dir

        #except:
            #pass

    output_file.close()

# AGE GENDER NET

def aggregate_probabs_ijba_agnet(preds_path, aggregated_preds_path):

    preds_results = open(preds_path).readlines()

    gender_preds_to_aggreg = []
    delimiter = ','

    video_dir = preds_results[0].split('\\\\')[6]

    output_file = open(aggregated_preds_path, 'w')

    line_counter = 0

    for pred in preds_results[0:]:
        #try:
            frame_dir = pred.split('\\\\')
            current_video_dir = frame_dir[6]
            line_counter += 1

            if (current_video_dir != video_dir) | (line_counter >= len(preds_results)):
                output_file.write(line[0] + ',')

                gender_sv = aggregate_gender_simple_vote_arr(gender_preds_to_aggreg)
                gender_am = get_from_list(aggregate_arithmetic_average(gender_preds_to_aggreg), gender_list)
                gender_gm = get_from_list(aggregate_geometric_average(gender_preds_to_aggreg), gender_list)
                gender_results = (gender_sv, gender_am, gender_gm)
                output_file.write(delimiter.join(gender_results) + delimiter)
                output_file.write('\n')

                gender_preds_to_aggreg = []

            gender_preds = []
            line = pred.split(',')
            gender_preds.append(float(line[2]))
            gender_preds.append(float(line[3]))
            gender_preds_to_aggreg.append(gender_preds)
            video_dir = current_video_dir

        #except:
            #pass

    output_file.close()

def calculate_accuracy_ijba(dataset_path, predictions_path, out_data):

    gender_map = generate_feature_map(dataset_path)
    age_map = generate_feature_map(dataset_path, 2)

    preds_results = open(predictions_path).readlines()
    delimiter = ','

    gender_err = 0

    for pred in preds_results[0:]:
        try:
            line = pred.split(',')
            current_frame = line[0]
            gender = line[1]

            if gender != gender_map.get(current_frame).lower():
                gender_err += 1
        except:
            pass

    output_file = open(out_data, 'w')
    output_file.write(str(gender_err) + delimiter)
    output_file.close()

def calculate_accuracy_aggreg_ijba(dataset_path, predictions_path, out_data):

    gender_map = generate_feature_map(dataset_path)

    preds_results = open(predictions_path).readlines()
    delimiter = ','

    sv_gender_err = 0
    am_gender_err = 0
    gm_gender_err = 0

    for pred in preds_results[0:]:
        #try:
            line = pred.split(',')
            current_dir = line[0]

            gender_sv = line[1]
            gender_am = line[2]
            gender_gm = line[3]

            if gender_sv != gender_map.get(current_dir).lower():
                sv_gender_err += 1
            if gender_am != gender_map.get(current_dir).lower():
                am_gender_err += 1
            if gender_gm != gender_map.get(current_dir).lower():
                gm_gender_err += 1
        #except:
            #pass

    output_file = open(out_data, 'w')
    err_gender_res = (str(sv_gender_err), str(am_gender_err), str(gm_gender_err))

    output_file.write(delimiter.join(err_gender_res))
    output_file.close()


if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))

    #KINECT

    #aggregate_probabs('C:/VKR/Kinect/Kinect_AgeGenderTF2192.csv', 'C:/VKR/Kinect/Aggreg_Kinect_AgeGenderTF2192.csv')
    #calculate_accuracy_aggreg('C:/Users/akharche/Desktop/VKR/Kinect/dataDirKinect.csv', 'C:/Users/akharche/Desktop/VKR/Kinect/TF2new/test.csv',
                                #'C:/Users/akharche/Desktop/VKR/Kinect/TF2new/out.csv')
    #calculate_accuracy('C:/Users/akharche/Desktop/VKR/Kinect/dataDirKinect.csv', 'C:/Users/akharche/Desktop/VKR/Kinect/TF2new/Kinect_AgeGenderTF2new.csv',
                        #'C:/Users/akharche/Desktop/VKR/Kinect/TF2new/err_kinect_tf2newNew.csv')

    # AGE_GENDER_NET

    #aggregate_probabs_agegendernet('C:/VKR/Kinect/Kinect_AgeGender_net_new.csv', 'C:/VKR/Kinect/Aggreg_Kinect_AgeGender_net_new.csv')
    #calculate_accuracy_aggreg('C:/VKR/Kinect/dataDirKinect.csv', 'C:/VKR/Kinect/Aggreg_Kinect_AgeGender_net_new.csv',
                          #'C:/VKR/Kinect/err_kinect_aggreg_agnet_new.csv')
    #calculate_accuracy('C:/VKR/Kinect/dataDirKinect.csv', 'C:/VKR/Kinect/Kinect_AgeGender_net_new.csv',
     #'C:/VKR/Kinect/err_kinect_agnet_new.csv')


    # IndianMovie

    #aggregate_probabs('C:/Users/akharche/Desktop/VKR/IndianMovie/TF2192/IM_AgeGenderTF2192.csv', 'C:/Users/akharche/Desktop/VKR/IndianMovie/TF2192/test1.csv')
    #calculate_accuracy_aggreg_im('C:/Users/akharche/Desktop/VKR/IndianMovie/datasetDir.csv', 'C:/Users/akharche/Desktop/VKR/IndianMovie/TF2192/test1.csv', 'C:/Users/akharche/Desktop/VKR/IndianMovie/TF2192/aggregacc.csv')
    #calculate_accuracy_im('C:/Users/akharche/Desktop/VKR/IndianMovie/datasetDir.csv', 'C:/Users/akharche/Desktop/VKR/IndianMovie/TF2new/IM_AgeGenderTF2new.csv', 'C:/Users/akharche/Desktop/VKR/IndianMovie/TF2192/err_indianmov_tf2newnew.csv')

    #aggregate_probabs('C:/Users/akharche/Desktop/VKR/IndianMovie/vgg16/IMresAge.csv',
                      #'C:/Users/akharche/Desktop/VKR/IndianMovie/vgg16/accuracy_IMresAge.csv')

    calculate_accuracy_aggreg_im('C:/Users/akharche/Desktop/VKR/IndianMovie/datasetDir.csv',
                          'C:/Users/akharche/Desktop/VKR/IndianMovie/vgg16/accuracy_IMresAge.csv',
                          'C:/Users/akharche/Desktop/VKR/IndianMovie/vgg16/accuracy_IMresAge1.csv')
    '''
    # AGE_GENDER_NET
    
    aggregate_probabs_agegendernet('C:/VKR/IndianMovie/IM_AgeGender_net.csv',
                                   'C:/VKR/IndianMovie/Aggreg_IM_AgeGender_net.csv')
    calculate_accuracy_aggreg_im('C:/VKR/IndianMovie/datasetDir.csv', 'C:/VKR/IndianMovie/Aggreg_IM_AgeGender_net.csv',
        'C:/VKR/IndianMovie/err_im_aggreg_agnet.csv')
    calculate_accuracy_im('C:/VKR/IndianMovie/datasetDir.csv', 'C:/VKR/IndianMovie/IM_AgeGender_net.csv',
          'C:/VKR/IndianMovie/err_im_agnet.csv')
    '''

    # IJBA
    '''
    make_dir('C:/VKR/IJBA/dir.csv', 'C:/VKR/IJBA/dir_new.csv')
    aggregate_probabs_ijba('C:/VKR/IJBA/IJBA_AgeGenderTF2192.csv', 'C:/VKR/IJBA/Aggreg_IJBA_AgeGenderTF2192.csv')
    calculate_accuracy_aggreg_ijba('C:/VKR/IJBA/dir_new.csv', 'C:/VKR/IJBA/Aggreg_IJBA_AgeGenderTF2192.csv',
                                'C:/VKR/IJBA/err_ijba_aggreg_tf2192.csv')
    calculate_accuracy_ijba('C:/VKR/IJBA/dir_new.csv', 'C:/VKR/IJBA/IJBA_AgeGenderTF2192.csv',
                                'C:/VKR/IJBA/err_ijba_tf2192.csv')
                                '''
    # AGE_GENDER_NET
    '''
    aggregate_probabs_ijba_agnet('C:/VKR/IJBA/IJBA_AgeGender_net.csv', 'C:/VKR/IJBA/Aggreg_IJBA_AgeGender_net.csv')
    calculate_accuracy_aggreg_ijba('C:/VKR/IJBA/dir_new.csv', 'C:/VKR/IJBA/Aggreg_IJBA_AgeGender_net.csv',
                                   'C:/VKR/IJBA/err_ijba_aggreg_agnet.csv')
    calculate_accuracy_ijba('C:/VKR/IJBA/dir_new.csv', 'C:/VKR/IJBA/IJBA_AgeGender_net.csv',
                            'C:/VKR/IJBA/err_ijba_agnet.csv')
    '''