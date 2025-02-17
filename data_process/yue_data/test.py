from cgitb import small
import json


def count_qa_type(data):
    count0, count1, count2, count3, count4, count5 = 0,0,0,0,0,0
    total_count = 0
    tmp = []
    key_list = []
    for key, value in data.items():
        key_list.append(key.split('_')[0])
        total_count += len(value['qa'])
        for q_a in value['qa']:
            question = q_a[0].lstrip()
            if question.lower().startswith('what'):
                tmp.append(question)
                count0 += 1
            elif question[:2].lower() == 'is':
                count1 += 1
            elif question[:3].lower() == 'how':
                count2 += 1
            elif question[:3].lower() == 'can':
                count3 += 1
            elif question.lower().startswith('which'):
                count4 += 1
            else:
                #tmp.append(question)
                count5 += 1
    return total_count, count0,count1,count2,count3,count4,count5


def count_qa_base(data):
    count0, count1, count2, count3, count4, count5 = 0,0,0,0,0,0
    total_count = 0
    tmp = []
    for item in data['questions']:
        question = item['question'].lstrip()
        if question[:4].lower() == 'what':
            count0 += 1
        elif question[:2].lower() == 'is':
            count1 += 1
        elif question[:3].lower() == 'how':
            count2 += 1
        elif question[:3].lower() == 'can':
            count3 += 1
        elif question[:5].lower() == 'which':
            count4 += 1
        else:
            tmp.append(question)
            count5 += 1
    return total_count, count0,count1,count2,count3,count4,count5
        


small_data_file = "/localscratch/zhan1624/embodied-generalist/data_process/yue_data/version4/yuesitqa_train4.json"
large_data_file = "/localscratch/zhan1624/embodied-generalist/data_process/yue_data/obj_post/obj_train.json"
base_file = "/egr/research-hlr2/zhan1624/leo_data/annotations/instruction/sqa3d/v1_balanced_questions_test_scannetv2.json"


with open(small_data_file) as f_small:
    small_data = json.load(f_small)

with open(large_data_file) as f_large:
    large_data = json.load(f_large)

with open(base_file) as f_base:
    base_data = json.load(f_base)

#count_qa_type(small_data)
count_qa_base(base_data)
print('yue')