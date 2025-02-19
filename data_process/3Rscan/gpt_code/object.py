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
write_file = "3Rscan/gpt_obj_qa/"

used_scene = os.listdir(write_file)

def split_list_average_n(origin_list, n):
    res = []
    for i in range(0, len(origin_list), n):
        res.append(origin_list[i: i+n])
    return res
        

def get_gpt(situation, new_dict):
    scene_info = json.dumps(new_dict)
    ques_prompt = """You need to generate at least 10 meaningful question-answer pairs based on the scene information. Ask questions about object types, and counting. The questions related to attributes are better be asked when multiple objects contain the same attributes and the answer can be specified based on spatial relations. You can also ask about spatial positions between me and other objects or spatial relations between objects by comparing the angles. Based on 'relations' in the scene graph, you can ask about other relations between objects. You need to provide the queried object. Do not consider the object's utility and affordance. Do not use the number of turn degrees or distance meters in the question and answer.  Do not use the IDs of the objects in the question and answer. The question answer pair should be following format:\nQ: <question>\nT: <queried object_id(s)>\nA: <Answer>. You can answer the question according to the queried object(s). If there is no information about the question, the <Answer> should be'unkown'.
There are several examples:
Q: What is the object closest to the left of me? T:lamp_1 Answer: a lamp.
Q: How many stools are on my left? T:stool_3 Answer: One.
Q: There are multiple chairs, what is the size of the chair left of me? T:chair_1, chair_2Answer: low chair.
Q: Is the cabinet far from me or the sofa far from me? Answer: sofa.
Q: How many black objects are to my right? Answer: Two, a towel and a toilet brush
Q: Where is the trash bin? Answer: Behind you.
Q: What color is the trash bin in front of me? Answer: black (one white left of me and one black in front of me)
Q: Is the mirror right of the shower curtain based on my standing position? Answer: Yes.
Q: Is the light on my right on or off?
Q: What is the object to the left of the white heater to my right? 
Q: Is there a picture to my right?
Q: Is the door in front of me the same color as the cabinet to my right? 
Q: The tv to your 11 o ' clock direction on ; true or false ?
Q: Can black objects are to my right be divided by three? 
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
        max_tokens=300,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

for id in tqdm(scene_ids):
    # if id in used_scene:
    #     continue
    qa_res = {}
    scene_id = id[:-5]
    if scene_id != "137a8158-1db5-2cc0-8003-31c12610471e":
        continue
    scene_file = "situated_meta/%s.json"%scene_id
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
