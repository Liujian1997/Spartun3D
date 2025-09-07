# -*- coding: utf-8 -*-
import json
import os
from tqdm import tqdm
import random
import itertools
import asyncio
from openai import AsyncOpenAI

random.seed(10)

# [CHANGED] 使用异步客户端（其余用法不变）
client = AsyncOpenAI(api_key="sk-c88b004dda1d489bb1635c4dbadf174d", base_url="http://localhost:3000/v1")

meta_file = "/data2/liujian/leo_data/scan_data/3RScan-base/3Rscan/situated_meta_download"
scene_ids = os.listdir(meta_file)

# type = "sit_cap"  # [CHANGED] 该变量未被使用且会覆盖内置 type，删除以避免风险
write_file = "/data2/liujian/leo_data/scan_data/3RScan-base/3Rscan/gpt_gen/gpt_afford_qa/"
used_scene = os.listdir(write_file)

def split_list_average_n(origin_list, n):
    res = []
    for i in range(0, len(origin_list), n):
        res.append(origin_list[i : i + n])
    return res
        

# [CHANGED] 使用新 SDK 的 async 调用
async def get_gpt(situation, new_dict):
    scene_info = json.dumps(new_dict)
    ques_prompt = """You need to generate 10 meaningful question-answer pairs based on the scene information. Ask questions about object affordance and object utility based on common sense. The answer should consider the best option that follows common sense knowledge and is closer to me. If I plan to go to some objects, other objects are blocking my way; please specify them. Do not use the number of turn degrees or distance meters in the question and answer. Do not use the IDs of the objects in the question and answer. You'd better make the answer specify the spatial position between you and the object. You need to provide the queried object. The question answer pair should be following format:\nQ: <question>\nT: <queried object_id(s)>\nA: <Answer>. You can answer the question according to the queried object(s). If there is no information about the question, the <Answer> should be'unknown'.''}]
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

    response = await client.chat.completions.create(
        model="/data2/liujian/checkpoints/Qwen2.5-32B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.strip()},
        ],
        max_tokens=350,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# [NEW] 信号量 + 重试
async def _call_get_gpt_with_limit(situation, new_example, semaphore, retries=2):
    for attempt in range(retries + 1):
        try:
            async with semaphore:
                return await get_gpt(situation, new_example)
        except Exception as e:
            if attempt >= retries:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))


# [NEW] 单 scene 内部并发，最后一次写文件
async def process_one_scene(scene_id, semaphore):
    scene_file = f"{meta_file}/{scene_id}.json"
    with open(scene_file) as f_in:
        scene_info = json.load(f_in)

    qa_res = {}
    futures_map = {}

    for key, example in scene_info.items():
        situation = example["situation"]
        position = example["position"]
        del example["situation"]
        del example["position"]

        qa_res[key] = {}
        new_example = {}
        dir_len_list = []
        directions = ["front", "right", "backwards", "left"]
        for d in directions:
            dir_len_list.append(len(example[d]))

        if sum(dir_len_list) > 60:
            sorted_id = sorted(range(len(dir_len_list)), key=lambda k: dir_len_list[k])
            quota = 60
            for c_idx, idx in enumerate(sorted_id):
                if c_idx == len(sorted_id) - 1:
                    sec_num = quota
                else:
                    sec_num = 15
                new_example[directions[idx]] = dict(
                    itertools.islice(example[directions[idx]].items(), sec_num)
                )
                quota -= len(new_example[directions[idx]])
        else:
            new_example = example.copy()

        # filter relations
        select_objs = []
        qa_res[key]["query"] = {}
        for d in new_example:
            select_objs += list(new_example[d].keys())
            qa_res[key]["query"][d] = list(new_example[d].keys())

        select_objs = [i.split("_")[1] for i in select_objs]
        select_objs.append(key)
        select_objs = list(set(select_objs))

        for d in new_example:
            for each_dir_obj in new_example[d]:
                new_rel = []
                relations = new_example[d][each_dir_obj]["relations"]
                if not relations:
                    continue
                for rel in relations:
                    if rel.split(" ")[-1].split("_")[1] in select_objs:
                        new_rel.append(rel)
                new_example[d][each_dir_obj]["relations"] = new_rel

        futures_map[key] = _call_get_gpt_with_limit(situation, new_example, semaphore)
        qa_res[key]["situation"] = situation
        qa_res[key]["pot"] = position

    keys = list(futures_map.keys())
    coros = [futures_map[k] for k in keys]
    results = await asyncio.gather(*coros, return_exceptions=False)
    for k, txt in zip(keys, results):
        qa_res[k]["obj_qa"] = txt

    tmp_path = os.path.join(write_file, f"{scene_id}.json.tmp")
    final_path = os.path.join(write_file, f"{scene_id}.json")
    with open(tmp_path, "w") as f_out:
        json.dump(qa_res, f_out, indent=4, ensure_ascii=False)
    os.replace(tmp_path, final_path)


# [NEW] 主入口：scene 串行，内部 key 并发
async def main():
    GPT_CONCURRENCY = int(os.getenv("GPT_CONCURRENCY", "8"))
    semaphore = asyncio.Semaphore(GPT_CONCURRENCY)

    for file_name in tqdm(scene_ids):
        if file_name in used_scene:
            continue
        scene_id = file_name[:-5]
        await process_one_scene(scene_id, semaphore)

if __name__ == "__main__":
    asyncio.run(main())