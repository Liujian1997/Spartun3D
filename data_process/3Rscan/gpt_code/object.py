# -*- coding: utf-8 -*-
import json
import os
from tqdm import tqdm
import random
import itertools
import asyncio
from openai import AsyncOpenAI  # ✅ 新：异步客户端

random.seed(10)
client = AsyncOpenAI(api_key="sk-c88b004dda1d489bb1635c4dbadf174d", base_url="http://localhost:3000/v1")

meta_file = "/data2/liujian/leo_data/scan_data/3RScan-base/3Rscan/situated_meta_download"
scene_ids = os.listdir(meta_file)

write_file = "/data2/liujian/leo_data/scan_data/3RScan-base/3Rscan/gpt_gen/gpt_obj_qa/"

used_scene = os.listdir(write_file)

def split_list_average_n(origin_list, n):
    res = []
    for i in range(0, len(origin_list), n):
        res.append(origin_list[i: i+n])
    return res
        

async def get_gpt(situation, new_dict):
    scene_info = json.dumps(new_dict)
    ques_prompt = """You need to generate at least 10 meaningful question-answer pairs based on the scene information. Ask questions about object types, and counting. The questions related to attributes are better be asked when multiple objects contain the same attributes and the answer can be specified based on spatial relations. You can also ask about spatial positions between me and other objects or spatial relations between objects by comparing the angles. Based on 'relations' in the scene graph, you can ask about other relations between objects. You need to provide the queried object. Do not consider the object's utility and affordance. Do not use the number of turn degrees or distance meters in the question and answer.  Do not use the IDs of the objects in the question and answer. The question answer pair should be following format:\nQ: <question>\nT: <queried object_id(s)>\nA: <Answer>. You can answer the question according to the queried object(s). If there is no information about the question, the <Answer> should be'unknown'.
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


    response = await client.chat.completions.create(
        model="/data2/liujian/checkpoints/Qwen2.5-32B-Instruct-AWQ",
        messages=[
                {"role": "system", "content": "You are a helpful assistant. "},
                {"role": "user", "content": prompt.strip()}
            ],
        max_tokens=5000,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# ✅ 并发控制 + 重试
async def _call_get_gpt_with_limit(situation, new_example, semaphore, retries=2):
    for attempt in range(retries + 1):
        try:
            async with semaphore:
                return await get_gpt(situation, new_example)
        except Exception as e:
            if attempt >= retries:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))

# ✅ 单 scene 处理：内部 key 并发，结束后原子写
async def process_one_scene(scene_id, semaphore):
    scene_file = os.path.join(meta_file, f"{scene_id}.json")
    with open(scene_file, "r") as f_in:
        scene_info = json.load(f_in)

    qa_res = {}
    futures_map = {}  # key -> coroutine

    for key, example in scene_info.items():
        situation = example['situation']
        position = example['position']

        del example['situation']
        del example['position']
        
        qa_res[key]= {}
        new_example = {}
        dir_len_list = []
        directions = ["front", "right", "backwards", "left"]
        for d in directions:  # 避免覆盖内置 dir()
            dir_len_list.append(len(example[d]))
        if sum(dir_len_list) > 60:
            sorted_id = sorted(range(len(dir_len_list)), key=lambda k: dir_len_list[k])
            quota = 60
            for c_idx, idx in enumerate(sorted_id):  # 避免覆盖内置 id()
                if c_idx == len(sorted_id)-1:
                    sec_num = quota
                else:
                    sec_num = 15
                new_example[directions[idx]] = dict(itertools.islice(example[directions[idx]].items(), sec_num))
                quota -= len(new_example[directions[idx]])
        else:
            new_example = example.copy()
        
        # filter relations
        select_objs = []
        qa_res[key]['query'] = {}
        for d in new_example:
            select_objs += list(new_example[d].keys())
            qa_res[key]['query'][d] = list(new_example[d].keys())

        select_objs = [i.split("_")[1] for i in select_objs]
        select_objs.append(key)
        select_objs = list(set(select_objs))

        for d in new_example:
            for each_dir_obj in new_example[d]:
                new_rel = []
                relations = new_example[d][each_dir_obj]['relations']
                if not relations:
                    continue
                for rel in relations:
                    if rel.split(" ")[-1].split("_")[1] in select_objs:
                        new_rel.append(rel)       
                new_example[d][each_dir_obj]['relations'] = new_rel

        # ✅ 登记并发任务
        futures_map[key] = _call_get_gpt_with_limit(situation, new_example, semaphore)

        # 附带信息
        qa_res[key]['situation'] = situation
        qa_res[key]['pot'] = position

    # ✅ 并发执行当前 scene 的所有请求
    keys = list(futures_map.keys())
    coros = [futures_map[k] for k in keys]
    results = await asyncio.gather(*coros, return_exceptions=False)
    for k, txt in zip(keys, results):
        qa_res[k]['obj_qa'] = txt

    # ✅ 原子写
    tmp_path = os.path.join(write_file, f"{scene_id}.json.tmp")
    final_path = os.path.join(write_file, f"{scene_id}.json")
    with open(tmp_path, "w") as f_out:
        json.dump(qa_res, f_out, indent=4, ensure_ascii=False)
    os.replace(tmp_path, final_path)

# ✅ 主入口：scene 串行，scene 内并发
async def main():
    GPT_CONCURRENCY = int(os.getenv("GPT_CONCURRENCY", "8"))
    semaphore = asyncio.Semaphore(GPT_CONCURRENCY)

    for file_name in tqdm(scene_ids):
        if file_name in used_scene:
            continue
        scene_id = file_name[:-5]  # 去掉 .json
        await process_one_scene(scene_id, semaphore)

if __name__ == "__main__":
    asyncio.run(main())