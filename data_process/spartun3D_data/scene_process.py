import json
import os
import torch
import numpy as np
import random
import math
from tqdm import tqdm

obj_file = "scans/3DSSG/objects.json"
relationship_file = "scans/3DSSG/relationships.json"
rscan_base = "leo_data/3RScan-base"
write_path = "data_process/3Rscan/situated_meta/"

#stop_object = ["otherstructure", "otherfurniture", "otherprop", "ceiling", "wall", "floor", "object"]
stop_object = ["ceiling", "wall", "floor", "object"]
stop_relation = [0,2,3,4,5,32]
#stop_relation = [0,32]
# specific_key = ["a644cb93-0ee5-2f66-9efb-b16adfb14eff", 
#                 "0cac768c-8d6f-2d13-8cc8-7ace156fc3e7",
#                 "6bde604d-9162-246f-8d7d-631c1e992162",
#                 "1d234002-e280-2b1a-8c8d-6aafb5b35a24",
#                 "0cac768a-8d6f-2d13-8dd3-3cbb7d916641",
#                 "77941464-cfdf-29cb-87f4-0465d3b9ab00"]
used_key = os.listdir(write_path)

#used_key = []
specific_key = []
    
with open(obj_file) as f_in1, open(relationship_file) as f_in2 :
    object_data = json.load(f_in1)
    relation_data = json.load(f_in2)


def get_angle(lineA,lineB):
    line1Y1 = lineA[0][1]
    line1X1 = lineA[0][0]
    line1Y2 = lineA[1][1]
    line1X2 = lineA[1][0]

    line2Y1 = lineB[0][1]
    line2X1 = lineB[0][0]
    line2Y2 = lineB[1][1]
    line2X2 = lineB[1][0]

    #calculate angle between pairs of lines
    angle1 = math.atan2(line1Y1-line1Y2,line1X1-line1X2)
    angle2 = math.atan2(line2Y1-line2Y2,line2X1-line2X2)
    front, backward, left, right = 0,0,0,0
    angle1_dg = (angle1) * 360 / (2*math.pi)
    angle2_dg = (angle2) * 360 / (2*math.pi)
    if angle1_dg < 0 :
        angle1_dg = angle1_dg + 360
    if angle2_dg < 0 :
        angle2_dg = angle2_dg + 360
    angle_degrees = angle1_dg - angle2_dg
    if angle_degrees > 0:
        if angle_degrees < 45:
            front = 1
        elif angle_degrees < 135:
            right = 1
        elif angle_degrees < 225 :
            backward=1
        elif angle_degrees < 315:
            left = 1
        else:
            front=1
    else:
        if angle_degrees > -45:
            front =1
        elif angle_degrees > -135:
            left = 1
        elif angle_degrees > -225:
            backward=1
        elif angle_degrees > -315:
            right=1
        else:
            front=1
        
    return front, backward, left, right, round(angle_degrees,2)


def get_distance_3D(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return round(distance, 1)

def get_att(object_attr):
    affordance,material,color, shape, state, lexical, size = "","","","","","", ""
    if "affordances" in object_attr:
        affordance = object_attr['affordances']
    if "material" in object_attr["attributes"]:
        material = object_attr["attributes"]["material"][0] + " "
    if "color" in object_attr["attributes"]:
        color = object_attr["attributes"]["color"][0]+" "
    if "shape" in object_attr["attributes"]:
        shape = object_attr["attributes"]["shape"][0]+" "
    if "state" in object_attr["attributes"]:
        state = object_attr["attributes"]["state"][0]+" "
    if "size" in object_attr["attributes"]:
        size = object_attr["attributes"]["size"][0]+" "
    if "lexical" in object_attr["attributes"]:
        lexical = object_attr["attributes"]["lexical"][0]+" "
    return affordance, material, color, shape, state, size


def get_distance_2D(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return round(distance, 1)

def convert_pc_to_box(obj_pc):
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
    aabb_min = np.array([xmin, ymin, zmin])
    aabb_max = np.array([xmax, ymax, zmax])
    return center, box_size, aabb_min, aabb_max

import numpy as np

def on_segment(p, q, r):
    """Given three collinear points p, q, r, the function checks if point q lies on line segment 'pr'"""
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    """To find the orientation of the ordered triplet (p, q, r).
    The function returns the following values:
    0 -> p, q and r are collinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

def do_intersect(p1, q1, p2, q2):
    """The function that returns True if the line segment 'p1q1' and 'p2q2' intersect."""
    # Find the four orientations needed for the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    # Doesn't fall in any of the above cases
    return False

def is_intersecting_rectangle(line_start, line_end, rect_bottom_left, rect_top_right):
    """Check if the line (line_start to line_end) intersects with the rectangle 
    defined by rect_bottom_left and rect_top_right."""
    rect_top_left = (rect_bottom_left[0], rect_top_right[1])
    rect_bottom_right = (rect_top_right[0], rect_bottom_left[1])

    # Check if the line intersects with any of the rectangle's sides
    if (do_intersect(line_start, line_end, rect_bottom_left, rect_top_left) or
            do_intersect(line_start, line_end, rect_top_left, rect_top_right) or
            do_intersect(line_start, line_end, rect_top_right, rect_bottom_right) or
            do_intersect(line_start, line_end, rect_bottom_right, rect_bottom_left)):
        return True

    return False


def get_rel(relations, id, inst_to_label):
    rel_list = []
    for rel in relations:
        if rel[-1] == "left":
            rel[-1] = "to the left of"
        elif rel[-1] == "right":
            rel[-1] = "to the right of"
        elif rel[-1] == "front":
            rel[-1] = "in front of"
        elif rel[-1] == "behind":
            rel[-1] = "behind of"           
        if rel[0] == int(id) and rel[2] not in stop_relation and inst_to_label[int(rel[1])] not in stop_object:
            rel_list.append(rel[-1] + " "+ inst_to_label[rel[1]]+"_"+str(rel[1]))
    return rel_list

def get_rel_reverse(relations, id, inst_to_label):
    rel_list_reverse = []
    for rel in relations:
        if rel[1] == int(id) and rel[2] not in stop_relation and inst_to_label[int(rel[0])] not in stop_object:
            rel_list_reverse.append(inst_to_label[rel[0]]+rel[-1] + " "+ inst_to_label[rel[1]])
    return rel_list_reverse


def get_block(refer_cord, referial_obj, targe_obj, allobj_dict, target_cord):
    block_list = []
    for id, obj in allobj_dict.items():
        if obj['label'] in stop_object or int(id) == referial_obj or id == targe_obj:
            continue
        obj_mask =  instance_labels == int(id)
        obj_cord, obj_size, aabb_min, aabb_max = convert_pc_to_box(pcds[obj_mask])
        block = is_intersecting_rectangle(refer_cord[:2], target_cord[:2], aabb_min[:2], aabb_max[:2])
        if block:
            if get_distance_2D(refer_cord, obj_cord)>=2 and obj_cord[-1]<3.5:
                block_list.append(obj['label']+"_"+obj['id'])
    return block_list

random.seed(10)
res_dict = {}
for item in tqdm(relation_data['scans']):
    scan_id = item['scan']
    # if scan_id in specific_key or scan_id+".json" in used_key:
    #     continue
    # if scan_id != "6e67e550-1209-2cd0-8294-7cc2564cf82c":
    #     continue

    scan_path = os.path.join(rscan_base, '3RScan-ours-align', scan_id)
    pcd_data = torch.load(os.path.join(scan_path, 'pcd-align.pth'))
    new_dict = {}
    for scan in object_data['scans']:
        if scan['scan'] == scan_id:
            objects = scan['objects']
            for obj in objects:
                new_dict[obj["id"]] = obj
            break
    relations = item['relationships']
    
    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2]
    inst_to_label = torch.load(os.path.join(scan_path, 'inst_to_label.pth'))

    if len(inst_to_label) <= 60:
        continue
    colors = colors / 127.5 - 1
    pcds = np.concatenate([points, colors], 1)

    dup_dict = {}
    for i_t_key, i_t_value in inst_to_label.items():
        if i_t_value not in dup_dict:
            dup_dict[i_t_value] = 1
        else:
            dup_dict[i_t_value] += 1

    refer_refer_dict = inst_to_label.copy()
    height_list = []
    for id, obj in new_dict.items():  
        obj_mask =  instance_labels == int(id)
        obj_cord, obj_size, aabb_min, aabb_max = convert_pc_to_box(pcds[obj_mask])
        height_list.append(obj_cord[-1])
    avg_height = sum(height_list)/len(height_list)
    for id, obj in new_dict.items():  
        obj_mask =  instance_labels == int(id)
        obj_cord, obj_size, aabb_min, aabb_max = convert_pc_to_box(pcds[obj_mask])
        height_list.append(obj_cord[-1])
        if obj_cord[-1] > avg_height+0.5:
            del refer_refer_dict[int(id)]

    final_obj = {}

    new_inst_to_label = {i_l_key:i_l_value for i_l_key, i_l_value in refer_refer_dict.items() if i_l_value not in stop_object}
    new_inst_to_label = {i_l_key:i_l_value for i_l_key, i_l_value in new_inst_to_label.items() if dup_dict[i_l_value] < 2}
    if len(new_inst_to_label) <= 2:
        continue

    
    for refer_obj, refer_name in new_inst_to_label.items():
        refer_mask =  instance_labels == refer_obj
        refer_cord, refer_size, refer_min, refer_max = convert_pc_to_box(pcds[refer_mask])
        refer_rel = get_rel(relations, refer_obj, inst_to_label)
        refer_affordance, refer_material, refer_color, refer_shape, refer_state, refer_size = get_att(new_dict[str(refer_obj)])

        stand_cord1 = [0,0,refer_cord[-1]]
        stand_cord2 = [0,0,refer_cord[-1]]
        refer_list = []
        if refer_cord[0] >= 0 and refer_cord[1] >= 0:
            # two options
            stand_cord1[0] = refer_cord[0]
            stand_cord1[1] = refer_min[1]
            stand_cord2[0] = refer_min[0]
            stand_cord2[1] = refer_cord[1]
        elif refer_cord[0] <= 0 and refer_cord[1] <= 0:
            stand_cord1[0] = refer_cord[0]
            stand_cord1[1] = refer_max[1]
            stand_cord2[0] = refer_max[0]
            stand_cord2[1] = refer_cord[1]
        elif refer_cord[0] <= 0 and refer_cord[1] >= 0:
            stand_cord1[0] = refer_cord[0]
            stand_cord1[1] = refer_min[1]
            stand_cord2[0] = refer_max[0]
            stand_cord2[1] = refer_cord[1]
        elif refer_cord[0] >= 0 and refer_cord[1] <= 0:
            stand_cord1[0] = refer_cord[0]
            stand_cord1[1] = refer_max[1]
            stand_cord2[0] = refer_min[0]
            stand_cord2[1] = refer_cord[1]
        refer_list.append(stand_cord1)
        refer_list.append(stand_cord2)
        stand_cord = random.sample([stand_cord1, stand_cord2],1)[0]

        if refer_state:
            referial_text = "I am standing besides "+refer_material+refer_color+refer_shape+refer_size+refer_name+" that is "+refer_state+". "
                
        else:
            referial_text = "I am standing besides "+refer_material+refer_color+refer_shape+refer_size+refer_name+". "

        key_id = str(refer_obj)
            
        final_obj[key_id] = {}
        # final_obj[key_id]["start"] = new_dict[str(referial_obj)]
        # final_obj[key_id]["start"]["relations"] = refer_rel
        # final_obj[key_id]["facing"] = new_dict[str(refer_obj)]
        final_obj[key_id]["left"] = {}
        final_obj[key_id]["front"] = {}
        final_obj[key_id]["backwards"] = {}
        final_obj[key_id]["right"] = {}
        # final_obj[key_id]["below"] = {}
        # final_obj[key_id]["up"] = {}
        final_obj[key_id]['situation'] = {}
        # other objects
        block_list = []
        for id, obj in new_dict.items():
            label_name = inst_to_label[int(id)]
            if label_name in stop_object or int(id) == refer_obj:
                    continue
            key = label_name+"_"+id
            
            obj_mask =  instance_labels == int(id)
            obj_cord, obj_size, aabb_min, aabb_max = convert_pc_to_box(pcds[obj_mask])
            #front, backwards, left, right, angles = get_angle((refer_cord[:2], refer_cord[:2]), (refer_cord[:2], obj_cord[:2]))
            front, backwards, left, right, angles = get_angle((refer_cord[:2],stand_cord[:2]), (obj_cord[:2], stand_cord[:2]))

            distance = get_distance_2D(stand_cord, obj_cord)

            block_list = get_block(refer_cord, refer_obj, id, new_dict, obj_cord)
            new_rel_list = get_rel(relations, id, inst_to_label)
            
            key_dict = {}
            key_dict[key] = {}
            key_dict[key]['distance'] = distance
            key_dict[key]['passby'] = block_list
            if 'affordances' in obj:
                key_dict[key]['affordances'] = obj['affordances']
            key_dict[key]['attributes'] = obj['attributes']
            if angles<0:
                key_dict[key]['angle'] = round(angles+360)
            else: 
                key_dict[key]['angle'] = angles

            if len(new_rel_list) > 3:
                key_dict[key]['relations'] = random.sample(new_rel_list,3)
            else:
                key_dict[key]['relations'] = new_rel_list
            ori_prompt = ""
            if distance < 0.5:
                affordance, material, color, shape, state, size = get_att(obj)
                if state:
                    ori_prompt += "There is a " + material + color + shape + size + \
                                                label_name + " that is "+state
                else: 
                    ori_prompt += "There is a " + material + color + shape + size + label_name
                    #ori_prompt += " above " + refer_name+". "
                if obj_cord[-1] < refer_cord[-1]:
                    ori_prompt += " below " + refer_name+". "
                else:
                    ori_prompt += " above " + refer_name+". "
                    #final_obj[key_id]["up"].update(key_dict)
            else:
                if front:
                    final_obj[key_id]["front"].update(key_dict)
                elif backwards:
                    final_obj[key_id]["backwards"].update(key_dict)
                elif left:
                    final_obj[key_id]["left"].update(key_dict)
                elif right:
                    final_obj[key_id]["right"].update(key_dict)
        
        final_obj[key_id]['situation'] = referial_text+" "+ori_prompt
        final_obj[key_id]['position'] = stand_cord
    # del final_obj[key_id]['start']
    # del final_obj[key_id]['facing']
    # del final_obj[key_id]['below']
    # del final_obj[key_id]['up']
    #with open(write_path+"/"+scan_id+".json", "w") as f_out:
        #json.dump(final_obj, f_out)
   

    







