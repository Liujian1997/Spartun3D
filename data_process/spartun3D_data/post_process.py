import json
import random
from tqdm import tqdm


data_file = "/localscratch/zhan1624/embodied-generalist/data_process/yue_data/obj_data/obj_train.json"
data_can_file1 = "/localscratch/zhan1624/embodied-generalist/data_process/yue_data/can_data/version1/can_train.json"
data_can_file2 = "/localscratch/zhan1624/embodied-generalist/data_process/yue_data/can_data/version2/afford_can_train.json"
write_file = "/localscratch/zhan1624/embodied-generalist/data_process/yue_data/obj_post/obj_train.json"
random.seed = 3

with open(data_file) as f_large:
    data = json.load(f_large)

can_data = {}
with open(data_can_file1) as f_can1, open(data_can_file2) as f_can2:
    can_data1 = json.load(f_can1)
    can_data2 = json.load(f_can2)
    can_data.update(can_data1)
    can_data.update(can_data2)


def sampling_qa(sample_id, set_num):
    if len(sample_id) > set_num:
        qa = [value['qa'][i] for i in random.sample(sample_id, set_num)]
    else:
        qa = [value['qa'][i] for i in sample_id]
    return qa



count0, count1, count2, count3, count4, count5 = 0,0,0,0,0,0
total_count = 0
what_list, is_list, how_list, can_list, which_list, other_list = [],[],[],[],[],[]

set_num = 2
new_data = {}
for key, value in tqdm(data.items()):
    if key == "137a8158-1db5-2cc0-8003-31c12610471e":
        print('yue')
    whatid, isid, howid, canid, whichid, otherid = [], [], [], [], [], []
    total_id = []
    for id, q_a in enumerate(value['qa']):
        question = q_a[0].lstrip()
        if question.lower().startswith('what'):
            whatid.append(id)
            what_list.append(question)
        elif question.lower().startswith('is'):
            isid.append(id)
            is_list.append(question)
        elif question.lower().startswith('how'):
            howid.append(id)
            how_list.append(question)
        elif question.lower().startswith('can'):
            canid.append(id)
            can_list.append(question)
        elif question.lower().startswith("which"):
            whichid.append(id)
            which_list.append(question)
        else:
            otherid.append(id)
            other_list.append(question)
    
    whatqa = sampling_qa(whatid, set_num)
    isqa = sampling_qa(isid, set_num)
    howqa = sampling_qa(howid, set_num)
    whichqa = sampling_qa(whichid, set_num)
    otherqa = sampling_qa(otherid, set_num)

    count0 += len(whatqa)
    count1 += len(isqa)
    count2 += len(howqa)
    count4 += len(whichqa)
    count5 += len(otherqa)

    total_qa = whatqa + isqa + howqa+ otherqa
    if key in can_data:
        canqa = can_data[key]['qa']
        total_qa = total_qa + canqa
        if canqa:
            total_qa = total_qa + random.sample(canqa, 1)
            count3 +=  1

    new_data[key] = value.copy()
    new_data[key]['qa'] = total_qa
        
# with open(write_file, 'w') as f_out:
#     json.dump(new_data, f_out)
print('yue')


