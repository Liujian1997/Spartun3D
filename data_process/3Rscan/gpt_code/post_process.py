import json
import os
from tqdm import tqdm
import numpy as np
import random
import torch
import math


def post_gpt(scene_file, write_file):
    #scene_id = "ba6fdaa8-a4c1-2dca-814b-4dd6cf0ed226"
    used_scene = os.listdir(write_file)
    stop_words = ['```python', 'question_answer_pairs = [','[',']', '```',"},", "}", 'questions_answers = [', 'qa_pairs = [']

    for scene in os.listdir(scene_file):
        if scene in used_scene:
            continue
        
        scene_id = scene[:-5]
     
        with open(scene_file+scene) as f_in:
            scene_data = json.load(f_in)
        new_dict = {}
        for key, example in scene_data.items():
            new_dict[key] = example.copy()
            new_dict[key]['obj_qa'] = []
            qa_list_raw = example['obj_qa'].split('\n\n')
            for each_qa in qa_list_raw:
                low_each_qa = each_qa.lower()
                if "sure" in low_each_qa or "certainly" in low_each_qa:
                    continue
                if "q:" not in low_each_qa or "a:" not in low_each_qa:
                    continue
                each_qa = each_qa.split('\n')
                for e_q in each_qa:
                    if e_q.startswith("Q:"):
                        question = e_q.split(':')[1].strip()
                    elif e_q.startswith("T:"):
                        query = e_q.split(':')[1].strip()
                    elif e_q.startswith("A:"):
                        answer = e_q.split(':')[1].strip()
                    else:
                        continue
                new_dict[key]['obj_qa'].append((question, query, answer))
        with open(write_file+scene_id+".json", 'w') as f_out:
            json.dump(new_dict, f_out, indent=4)


def collect_sitqa():
    gpt_folder = "3Rscan/gpt_obj_clean/"
    json_files = os.listdir(gpt_folder)
    all_data = {}
    total = 0
    for file in tqdm(json_files):
        scene_id = file[:-5]
        with open(gpt_folder+file) as f_in:
            data = json.load(f_in)
        for key, value in data.items():
            for qa_id, qa in enumerate(value['qa']):
                new_qa = {key.lower(): value for key, value in qa.items()}
                all_data[scene_id+"_"+key+"_"+str(qa_id)] = new_qa
                all_data[scene_id+"_"+key+"_"+str(qa_id)]['situation'] = value['situation']
    with open('spartun3D_data/sitqa.json', 'w') as f_out:
        json.dump(all_data, f_out)


def convert_pc_to_box(obj_pc):
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
    aabb_min = [xmin, ymin, zmin]
    aabb_max = [xmax, ymax, zmax]
    return center, box_size, aabb_min, aabb_max


def axis_angle_to_quaternion(axis, angle):
    """
    Convert an axis-angle rotation to a quaternion.
    
    Parameters:
    axis (tuple or list): The axis of rotation (x, y, z)
    angle (float): The rotation angle in radians
    
    Returns:
    tuple: The quaternion (w, x, y, z)
    """
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    half_angle = angle / 2
    w = np.cos(half_angle)
    x, y, z = axis * np.sin(half_angle)
    return [x, y, z, w]


def calculate_rot(vec1_list,vec2_list):
    vec1 = np.array([0,0,1])
    if vec1_list[0] == vec2_list[0]:
        vec1 = np.array([1,0,1])
        vec2 = np.array([vec1_list[0], 0, 1])
    if vec1_list[1] == vec2_list[1]:
        vec1 = np.array([1,0,1])
        vec2 = np.array([0, vec1_list[1], 1])
    
    # Normalize the vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Calculate the dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate the angle in radians
    angle = np.arccos(dot_product)

    half_angle = angle / 2
    w = np.cos(half_angle)
    x, y, z = vec1 * np.sin(half_angle)
    return [x, y, z, w]


def sentence_process(sent):
    sent_list = sent.split(".")
    left_sent = []
    right_sent = []
    back_sent = []
    front_sent = []
    for each_sent in sent_list:
        tmp_sent = each_sent.lower()
        if each_sent:
            each_sent += "."
        if "left" in tmp_sent[:15]:
            left_sent.append(each_sent)
        elif "right" in tmp_sent[:15]:
            right_sent.append(each_sent)
        elif "back" in tmp_sent[:15] or "behind" in tmp_sent[:15]:
            back_sent.append(each_sent)
        elif "front" in tmp_sent[:15]:
            front_sent.append(each_sent)
    
    if not left_sent:
        left_sent = ["To my left, there is no object."]
    if not front_sent:
        front_sent = ["In front of me, there is no object."]
    if not right_sent:
        right_sent = ['To my right, there is no object.']
    if not back_sent:
        back_sent = ['Behind of me, there is no object.']

    function_sent = ""
    if "features"in sent_list[-2] or "function" in sent_list[-2] or "activity" in sent_list[-2] or "activities" in sent_list[-2]:
        function_sent = sent_list[-2]
    return " ".join([" ".join(front_sent), " ".join(right_sent), " ".join(back_sent), " ".join(left_sent)])+" "+function_sent


def collect_sitcap():
    random.seed = 10
    gpt_folder = "data_process/3Rscan/gpt_cap/"
    rscan_base = "leo_data/3RScan-base"

    train_file = "leo_data/annotations/alignment/scene_caption/3rscan_scenecap_train.json"
    val_file = "leo_data/annotations/alignment/scene_caption/3rscan_scenecap_val.json"

    with open(train_file) as f_train, open(val_file) as f_val:
        train_scans = list(json.load(f_train).keys())
        val_scans = list(json.load(f_val).keys())


    json_files = os.listdir(gpt_folder)

    all_data = {}
    train_data = {}
    val_data = {}
    test_data = {}

    total = 0
    token_length = []
    specific_id = ["6bde604b-9162-246f-8fb2-2dea80e7fb4c","a895258d-9035-254b-8e61-2307a9926e62",
                   "ba6fdab4-a4c1-2dca-83e2-f5878fdf688a","c12890e3-d3df-2d0d-87cf-a5510bc39c3a",
                   "f2c76fe7-2239-29d0-84f5-144c30fd7451", "63b87cf1-ef3f-28f2-871a-c1551f129ce6"]
    for file in tqdm(json_files):
        scene_id = file[:-5]
        if scene_id in specific_id:
            continue
        ### modify situation
        scan_path = os.path.join(rscan_base, '3RScan-ours-align', scene_id)
        pcd_data = torch.load(os.path.join(scan_path, 'pcd-align.pth'))
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2]
        inst_to_label = torch.load(os.path.join(scan_path, 'inst_to_label.pth'))
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)

        with open("3Rscan/situated_meta/%s.json"%scene_id) as f_meta:
            meta_data = json.load(f_meta)

        
        with open(gpt_folder+file) as f_in:
            gpt_data = json.load(f_in)
        for key, value in gpt_data.items():
            if not value['query']['left'] or not value['query']['right'] or not value['query']['front'] or not value['query']['backwards']:
                value['caption'] = sentence_process(value['caption'])
            # other relations:
            relation_obj = []
            for each_dir  in value['query']:
                for dir_obj in value['query'][each_dir]:
                    rel_list = meta_data[key][each_dir][dir_obj]['relations']
                    if rel_list:
                        for each_rel in rel_list:
                            relation_obj.append(each_rel.split(" ")[-1])

            refer_mask = instance_labels  == int(key)
            refer_cord, refer_size, refer_min, refer_max = convert_pc_to_box(pcds[refer_mask])
            
            ### initial rot
            rot = calculate_rot(value['pot'], (refer_cord))        
            objects_list = []
            distinct_obj = {}
            for ori_key, ori_value in meta_data[key].items():
                if ori_key == "situation" or ori_key == "position":
                    continue
                for obj_key, obj_value in ori_value.items():
                    objects_list.append((ori_key, obj_key, obj_value['distance']))
                    if obj_key.split("_")[0] not in distinct_obj:
                        distinct_obj[obj_key.split("_")[0]] = 1
                    else:
                        distinct_obj[obj_key.split("_")[0]] += 1
            if len(objects_list) == 0:
                continue
            avg_dis = sum(list(zip(*objects_list))[-1])/len(objects_list)
            while True:
                refer_obj = random.choice(objects_list)
                if refer_obj[-1] <= avg_dis+0.5 and distinct_obj[refer_obj[1].split('_')[0]]<=3:
                    break
            refer_text = "there is a " +refer_obj[1].split("_")[0]+" on your "+refer_obj[0]+"."
            all_data[scene_id+"_"+key] = {}
            all_data[scene_id+"_"+key]['pos'] = value['pot']
            all_data[scene_id+"_"+key]['rot'] = rot
            caption = value['caption'].replace('\n\n', " ")
            all_data[scene_id+"_"+key]['caption'] = ". ".join(caption.split(". "))
            #all_data[scene_id+"_"+key]['caption'] = "Your standing coorindate is "+ str(round(stand_cord[0],2)) + ", " + str(round(stand_cord[1],2)) + ", " + str(round(stand_cord[2],2))+"."
            token_length.append(len(value['caption'].split(" ")))
            all_data[scene_id+"_"+key]['situation'] = "You are standing beside "+ inst_to_label[int(key)]+" while "+\
                                                      refer_text + value['situation'].split(".")[1]
            all_data[scene_id+"_"+key]['query'] = value['query']
            all_data[scene_id+"_"+key]['relation'] = relation_obj
            all_data[scene_id+"_"+key]['refer'] = refer_obj[1].split("_")[1]
            if scene_id in train_scans:
                train_data[scene_id+"_"+key] = all_data[scene_id+"_"+key]
            elif scene_id in val_scans:
                val_data[scene_id+"_"+key] = all_data[scene_id+"_"+key]
            else:
                test_data[scene_id+"_"+key] = all_data[scene_id+"_"+key]
            total += 1

    with open('spartun3D_data/sitcap_train.json', 'w') as f_out:
        json.dump(train_data, f_out)
    
    with open('spartun3D_data/sitcap_val.json', 'w') as f_out:
        json.dump(val_data, f_out)
    
    with open('spartun3D_data/sitcap_test.json', 'w') as f_out:
        json.dump(test_data, f_out)


def collect_sitqa(process_type):
    random.seed = 10
    gpt_folder = f"/3Rscan/gpt_%s_clean/"%process_type
    rscan_base = "leo_data/3RScan-base"

    train_file = "leo_data/annotations/alignment/scene_caption/3rscan_scenecap_train.json"
    val_file = "leo_data/annotations/alignment/scene_caption/3rscan_scenecap_val.json"

    with open(train_file) as f_train, open(val_file) as f_val:
        train_scans = list(json.load(f_train).keys())
        val_scans = list(json.load(f_val).keys())


    json_files = os.listdir(gpt_folder)

    all_data = {}
    train_data = {}
    val_data = {}
    test_data = {}

    total = 0
    token_length = []
    specific_id = ["6bde604b-9162-246f-8fb2-2dea80e7fb4c","a895258d-9035-254b-8e61-2307a9926e62",
                   "ba6fdab4-a4c1-2dca-83e2-f5878fdf688a","c12890e3-d3df-2d0d-87cf-a5510bc39c3a",
                   "f2c76fe7-2239-29d0-84f5-144c30fd7451", "63b87cf1-ef3f-28f2-871a-c1551f129ce6"]
    for file in tqdm(json_files):
        # if len(all_data)> 300:
        #     break
        scene_id = file[:-5]
        # if scene_id != "f62fd5fd-9a3f-2f44-883a-1e5cf819608e":
        #     continue
        if scene_id in specific_id:
            continue
        ### modify situation
        scan_path = os.path.join(rscan_base, '3RScan-ours-align', scene_id)
        pcd_data = torch.load(os.path.join(scan_path, 'pcd-align.pth'))
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2]
        inst_to_label = torch.load(os.path.join(scan_path, 'inst_to_label.pth'))
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)

        with open("situated_meta/%s.json"%scene_id) as f_meta:
            meta_data = json.load(f_meta)
     
        with open(gpt_folder+file) as f_in:
            gpt_data = json.load(f_in)
        for key, value in gpt_data.items():
            refer_mask = instance_labels  == int(key)
            refer_cord, refer_size, refer_min, refer_max = convert_pc_to_box(pcds[refer_mask])
            
            ### initial rot
            rot = calculate_rot(value['pot'], (refer_cord))        
            objects_list = []
            distinct_obj = {}
            for ori_key, ori_value in meta_data[key].items():
                if ori_key == "situation" or ori_key == "position":
                    continue
                for obj_key, obj_value in ori_value.items():
                    objects_list.append((ori_key, obj_key, obj_value['distance']))
                    if obj_key.split("_")[0] not in distinct_obj:
                        distinct_obj[obj_key.split("_")[0]] = 1
                    else:
                        distinct_obj[obj_key.split("_")[0]] += 1
            if len(objects_list) == 0:
                continue
            avg_dis = sum(list(zip(*objects_list))[-1])/len(objects_list)
            while True:
                refer_obj = random.choice(objects_list)
                if refer_obj[-1] <= avg_dis+0.5 and distinct_obj[refer_obj[1].split('_')[0]]<=3:
                    break
            refer_text = "there is a " +refer_obj[1].split("_")[0]+" on your "+refer_obj[0]+"."
            all_data[scene_id+"_"+key] = {}
            all_data[scene_id+"_"+key]['pos'] = value['pot']
            all_data[scene_id+"_"+key]['rot'] = rot
            all_data[scene_id+"_"+key]['qa'] = value['obj_qa']
            all_data[scene_id+"_"+key]['situation'] = "You are standing beside "+ inst_to_label[int(key)]+" while "+\
                                                      refer_text + value['situation'].split(".")[1]
            all_data[scene_id+"_"+key]['query'] = value['query']
            all_data[scene_id+"_"+key]['refer'] = refer_obj[1].split("_")[1]
            if scene_id in train_scans:
                train_data[scene_id+"_"+key] = all_data[scene_id+"_"+key]
            elif scene_id in val_scans:
                val_data[scene_id+"_"+key] = all_data[scene_id+"_"+key]
            else:
                test_data[scene_id+"_"+key] = all_data[scene_id+"_"+key]
            total += 1

    with open(f'spartun3D_data/%s_data/%s_train.json'%(process_type,process_type), 'w') as f_out:
        json.dump(train_data, f_out)
    
    with open(f'spartun3D_data/%s_data/%s_val.json'%(process_type,process_type), 'w') as f_out:
        json.dump(val_data, f_out)
    
    with open('spartun3D_data/%s_data/%s_test.json'%(process_type,process_type), 'w') as f_out:
        json.dump(test_data, f_out)




scene_file_obj = "3Rscan/gpt_afford_can/"
write_file_obj = "3Rscan/gpt_afford_can_clean/"
#post_gpt(scene_file_obj, write_file_obj)

scene_file_afford = "3Rscan/gpt_afford_qa/"
write_file_afford = "3Rscan/gpt_afford_clean/"
# post_gpt(scene_file_afford, write_file_afford)

scene_file_planning = "3Rscan/gpt_planning/"
write_file_planning = "3Rscan/gpt_plan_clean/"
post_gpt(scene_file_planning, write_file_planning)

#collect_total()
#collect_sitcap()
qa_type = "plan"
collect_sitqa(qa_type)