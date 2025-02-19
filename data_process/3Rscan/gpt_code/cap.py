import openai
import json
import os
from tqdm import tqdm
import random

random.seed = 10
openai.api_key = ''
meta_file = "3Rscan/situated_meta"
scene_ids = os.listdir(meta_file)

type = "sit_cap"
write_file = "3Rscan/gpt_cap/"

used_scene = os.listdir(write_file)

def split_list_average_n(origin_list, n):
    res = []
    for i in range(0, len(origin_list), n):
        res.append(origin_list[i: i+n])
    return res


def get_gpt(direction, situation, new_dict):
    scene_info = json.dumps(new_dict)
    ques_prompt = """Describe the objects on your %s from the lowest to highest angle. Describe the scene using commonsense, such as how objects can be used by humans and human activities in the scene.  The description should conform to the given scene information. You need to describe each object in the scene. Your summary must be one paragraph, not exceeding 25 words. Don't use IDs of the objects in the summary. Don't use turn degrees or distance meters in the summary. 
    """%direction
    prompt = """%s The scene contains some objects, which compose a scene graph in JSON format with four keys: "left", "right", "front", "backwards", indicating objects in the corresponding direction. Each entity in the scene graph denotes an object instance with a class label and an object ID. The 'distance' indicates the meters between the object and me. The 'angle' represents the degrees compared to my current direction, where my direction in front is 0 degrees, and larger angles are to the right of objects with smaller angles. The 'affordance' is the motion activity related to this object. The 'attributes' describe the object's characteristics, such as 'color' and 'material'. The 'relations' describe the spatial relationships with other objects. The 'passby' indicates which objects would appear in my path if I walk toward them.

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
        max_tokens=60,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

for id in tqdm(scene_ids):
    # if id in used_scene:
    #     continue
    qa_res = {}
    scene_id = id[:-5]
    # if scene_id != "f62fd5fd-9a3f-2f44-883a-1e5cf819608e":
    #     continue 
    scene_file = "3Rscan/situated_meta/%s.json"%scene_id
    with open(scene_file) as f_in:
        scene_info = json.load(f_in)
    count = 0
    for key, example in scene_info.items():
        # if count >10:
        #     break
        situation = example['situation']
        del example['situation']
        
        left_cap_list = []
        sec_num = 10
        
        qa_res[key]= {}
        qa_res[key]['query'] = {}

        for dir in ["front", "right", "backwards", "left"]:
            qa_res[key]['query'][dir] = []
            qa_res[key]['query'][dir+"_cap"] = []
            if len(example[dir]) == 0:
                continue
            all_objs = list(example[dir])
            random.shuffle(all_objs)
            obj_list = split_list_average_n(all_objs, sec_num)
            if len(obj_list)>=2 and len(obj_list[-1])< sec_num/2:
                tmp_obj = obj_list[-2]+obj_list[-1]
                obj_list = obj_list[:-2]
                obj_list.append(tmp_obj)
            new_dict = {}
            for obj in obj_list:
                new_dict[dir] = {}
                for l_o in obj:
                    new_dict[dir][l_o] = example[dir][l_o]
                cap = get_gpt(dir, situation, new_dict)

                # generte the generated questions and answers
                qa_res[key]['query'][dir].append(obj)
                qa_res[key]['query'][dir+"_cap"].append(cap)
            qa_res[key]['situation'] = situation
            qa_res[key]['pot'] = example['position']

    with open(write_file+scene_id+".json", 'w') as f_out:
        json.dump(qa_res, f_out, indent=4)
