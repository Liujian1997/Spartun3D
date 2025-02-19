import openai
import json
import os
from tqdm import tqdm
import random
import itertools

random.seed = 10
openai.api_key = ''
meta_file = "3Rscan/situated_meta"
scene_ids = os.listdir(meta_file)

type = "sit_cap"
write_file = "3Rscan/gpt_afford_qa/"

used_scene = os.listdir(write_file)

def split_list_average_n(origin_list, n):
    res = []
    for i in range(0, len(origin_list), n):
        res.append(origin_list[i: i+n])
    return res
        

def get_gpt(situation, new_dict):
    scene_info = json.dumps(new_dict)
    ques_prompt = """You need to generate 10 meaningful question-answer pairs based on the scene information. Ask questions about object affordance and object utility based on common sense. The answer should consider the best option that follows common sense knowledge and is closer to me. If I plan to go to some objects, other objects are blocking my way; please specify them. Do not use the number of turn degrees or distance meters in the question and answer. Do not use the IDs of the objects in the question and answer. You'd better make the answer specify the spatial position between you and the object. You need to provide the queried object. The question answer pair should be following format:\nQ: <question>\nT: <queried object_id(s)>\nA: <Answer>. You can answer the question according to the queried object(s). If there is no information about the question, the <Answer> should be'unkown'.''}]
There are several examples:
Q: Where should I go to quickly put something down?A: You can use the chair in front of you.
Q: I want to read a book. Should I walk to the window or sit in the chair? A: Sit in the chair, which is closer.
Q: If I want to reach the kitchen counter, what object will be passed by? Answer: tables with chairs.
Q: What should I do if I want to cook? A: You can go to the kitchen area, but be careful about the tables and chairs you will pass by.
    """

    prompt = """%s The scene contains some objects, which compose a scene graph in JSON format with four keys: "left", "right", "front", "backwards", indicating objects in the corresponding direction. Each entity in the scene graph denotes an object instance with a class label and an object ID. The 'distance' indicates the meters between the object and me. The 'angle' represents the degrees compared to my current direction, where my direction in front is 0 degrees. The larger angles are to my right means further right. The larger angles to my backward mean further back. The larger angles are to my left means further left. The 'affordance' is the motion activity related to this object. The 'attributes' describe the object's characteristics, such as 'color' and 'material'. The 'relations' describe the spatial relationships with other objects. The 'passby' indicates which objects would appear in my path if I walk toward them.

    For example, from the scene graph "Left": {{"table_8": {"distance": 2.6, "passby": ["chair_21"], "affordances": ["placing items on", "cleaning", "carrying"], "attributes": {{"color":"red"}}, "angle": 257.48, "relations": ["close by chair_36", "close by chair_27", "close by chair_19"]}}}. We can know that on my right 257.48 degrees and 2.6 meters, there is a table_8 that is close by chair_36. You can place items on or clean or carry this table_8. If you go to table_8, you could pass by chair_21.

    %s
    Here is the scene information in JSON format: %s
    """%(situation, ques_prompt, scene_info)


    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
                {"role": "system", "content": "You are a helpful assistant. "},
                {"role": "user", "content": prompt.strip()}
            ],
        max_tokens=350,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

for id in tqdm(scene_ids):
    if id in used_scene:
        continue
    qa_res = {}
    scene_id = id[:-5]
    # if scene_id != "20c99392-698f-29c5-8439-54bec948ecb1":
    #     continue
    scene_file = "3Rscan/situated_meta/%s.json"%scene_id
    with open(scene_file) as f_in:
        scene_info = json.load(f_in)
    count = 0
    for key, example in scene_info.items():
        # if count >10:
        #     break
        situation = example['situation']
        position = example['position']

        del example['situation']
        del example['position']
        
        left_cap_list = []
        qa_res[key]= {}
        new_example = {}
        dir_len_list = []
        directions = ["front", "right", "backwards", "left"]
        for dir in directions:
            dir_len_list.append(len(example[dir]))
        if sum(dir_len_list) > 60:
            sorted_id = sorted(range(len(dir_len_list)), key=lambda k: dir_len_list[k])
            count = 60
            for c_id, id in enumerate(sorted_id):
                if c_id == len(sorted_id)-1:
                    sec_num = count
                else:
                    sec_num = 15
                new_example[directions[id]] = dict(itertools.islice(example[directions[id]].items(), sec_num))
                count -= len(new_example[directions[id]])
        else:
            new_example = example.copy()
        
        # filter relations
        select_objs = []
        qa_res[key]['query'] = {}
        for dir in new_example:
            select_objs += list(new_example[dir].keys())
            qa_res[key]['query'][dir] = list(new_example[dir].keys())

        select_objs = [i.split("_")[1] for i in select_objs]
        select_objs.append(key)
        select_objs = list(set(select_objs))

        for dir in new_example:
            for each_dir_obj in new_example[dir]:
                new_rel = []
                relations = new_example[dir][each_dir_obj]['relations']
                if not relations:
                    continue
                for rel in relations:
                    if rel.split(" ")[-1].split("_")[1] in select_objs:
                        new_rel.append(rel)       
                new_example[dir][each_dir_obj]['relations'] = new_rel
        qa = get_gpt(situation, new_example)


        
        qa_res[key]['obj_qa'] = qa   
        qa_res[key]['situation'] = situation
        qa_res[key]['pot'] = position

    with open(write_file+scene_id+".json", 'w') as f_out:
        json.dump(qa_res, f_out, indent=4)
