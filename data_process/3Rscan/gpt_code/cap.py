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

write_file = "/data2/liujian/leo_data/scan_data/3RScan-base/3Rscan/gpt_gen/gpt_cap/"

used_scene = os.listdir(write_file)

def split_list_average_n(origin_list, n):
    res = []
    for i in range(0, len(origin_list), n):
        res.append(origin_list[i: i+n])
    return res


async def get_gpt(direction, situation, new_dict):
    scene_info = json.dumps(new_dict)
    ques_prompt = """Describe the objects on your %s from the lowest to highest angle. Describe the scene using commonsense, such as how objects can be used by humans and human activities in the scene.  The description should conform to the given scene information. You need to describe each object in the scene. Your summary must be one paragraph, not exceeding 25 words. Don't use IDs of the objects in the summary. Don't use turn degrees or distance meters in the summary. 
    """%direction
    prompt = """%s The scene contains some objects, which compose a scene graph in JSON format with four keys: "left", "right", "front", "backwards", indicating objects in the corresponding direction. Each entity in the scene graph denotes an object instance with a class label and an object ID. The 'distance' indicates the meters between the object and me. The 'angle' represents the degrees compared to my current direction, where my direction in front is 0 degrees, and larger angles are to the right of objects with smaller angles. The 'affordance' is the motion activity related to this object. The 'attributes' describe the object's characteristics, such as 'color' and 'material'. The 'relations' describe the spatial relationships with other objects. The 'passby' indicates which objects would appear in my path if I walk toward them.

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

async def _call_get_gpt_with_limit(direction, situation, new_dict, semaphore, retries=2):
    for attempt in range(retries + 1):
        try:
            async with semaphore:
                return await get_gpt(direction, situation, new_dict)
        except Exception as e:
            if attempt >= retries:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))

# ===== 单个scene处理（scene内并发）=====
async def process_one_scene(scene_id, semaphore):
    scene_file = os.path.join(meta_file, f"{scene_id}.json")
    with open(scene_file, "r", encoding="utf-8") as f_in:
        scene_info = json.load(f_in)

    qa_res = {}
    futures = []  # (key, dir_name, idx, obj_chunk, coro)

    for key, example in scene_info.items():
        situation = example['situation']
        # position 只在 cap 里作为附带信息使用（原脚本里写入了 pot）
        position = example.get('position')

        # 移除不参与提示词的字段
        example = {k: v for k, v in example.items() if k not in ("situation",)}

        qa_res[key] = {}
        qa_res[key]['query'] = {}
        # 原脚本：每个方向按 sec_num=10 切块，并发请求
        sec_num = 10
        directions = ["front", "right", "backwards", "left"]
        for d in directions:
            qa_res[key]['query'][d] = []
            qa_res[key]['query'][d+"_cap"] = []
            if len(example.get(d, {})) == 0:
                continue

            all_objs = list(example[d])
            random.shuffle(all_objs)
            obj_list = split_list_average_n(all_objs, sec_num)
            if len(obj_list) >= 2 and len(obj_list[-1]) < sec_num / 2:
                tmp_obj = obj_list[-2] + obj_list[-1]
                obj_list = obj_list[:-2]
                obj_list.append(tmp_obj)

            # 先占位，保证返回后能按 idx 放
            qa_res[key]['query'][d] = [None] * len(obj_list)
            qa_res[key]['query'][d + "_cap"] = [None] * len(obj_list)

            for idx, obj_chunk in enumerate(obj_list):
                # 构造本批次 new_dict
                new_dict = {d: {}}
                for l_o in obj_chunk:
                    new_dict[d][l_o] = example[d][l_o]

                coro = _call_get_gpt_with_limit(d, situation, new_dict, semaphore)
                futures.append((key, d, idx, obj_chunk, coro))

        qa_res[key]['situation'] = situation
        qa_res[key]['pot'] = position  # 与你原脚本保持一致的字段名

    # 并发执行所有登记的任务
    coros = [it[-1] for it in futures]
    results = await asyncio.gather(*coros, return_exceptions=False)

    # 回填结果（按原结构）
    for (key, d, idx, obj_chunk, _), cap in zip(futures, results):
        qa_res[key]['query'][d][idx] = obj_chunk
        qa_res[key]['query'][d + "_cap"][idx] = cap

    # 原子写文件
    tmp_path = os.path.join(write_file, f"{scene_id}.json.tmp")
    final_path = os.path.join(write_file, f"{scene_id}.json")
    with open(tmp_path, "w", encoding="utf-8") as f_out:
        json.dump(qa_res, f_out, indent=2, ensure_ascii=False)
    os.replace(tmp_path, final_path)

# ===== 主入口：scene 串行，内部并发 =====
async def main():
    GPT_CONCURRENCY = int(os.getenv("GPT_CONCURRENCY", "8"))
    semaphore = asyncio.Semaphore(GPT_CONCURRENCY)

    for file_name in tqdm(scene_ids):
        # 如果想跳过已生成的 scene，取消注释：
        # if file_name in used_scene:
        #     continue
        scene_id = file_name[:-5]  # 去掉 .json
        await process_one_scene(scene_id, semaphore)

if __name__ == "__main__":
    asyncio.run(main())