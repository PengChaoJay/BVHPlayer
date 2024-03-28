import json
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

import asyncio
import websockets

class HierarchyParser(object):

    def __init__(self, bvh_file_path):
        self.lines = self.get_hierarchy_lines(bvh_file_path)
        self.line_number = 0

        self.root_position_channels = []
        self.joint_rotation_channels = []

        self.joint_names = []
        self.joint_parents = []
        self.joint_offsets = []

    def get_hierarchy_lines(self, bvh_file_path):
        hierarchy_lines = []
        for line in open(bvh_file_path, 'r'):
            line = line.strip()
            if line.startswith('MOTION'):
                break
            else:
                hierarchy_lines.append(line)

        return hierarchy_lines
    
    def parse_offset(self, line):
        return [float(x) for x in line.split()[1:]]
    
    def parse_channels(self, line):
        return [x for x in line.split()[2:]]

    def parse_root(self, parent=-1):
        self.joint_parents.append(parent)

        self.joint_names.append(self.lines[self.line_number].split()[1])
        self.line_number += 2

        if self.lines[self.line_number].startswith('OFFSET'):
            self.joint_offsets.append(self.parse_offset(self.lines[self.line_number]))
        else:
            print('cannot find root offset')
        self.line_number += 1

        if self.lines[self.line_number].startswith('CHANNELS'):
            channels = self.parse_channels(self.lines[self.line_number])
            if self.lines[self.line_number].split()[1] == '3':
                self.joint_rotation_channels.append((channels[0], channels[1], channels[2]))
            elif self.lines[self.line_number].split()[1] == '6':
                self.root_position_channels.append((channels[0], channels[1], channels[2]))
                self.joint_rotation_channels.append((channels[3], channels[4], channels[5]))
        else:
            print('cannot find root channels')
        self.line_number += 1

        while self.lines[self.line_number].startswith('JOINT'):
            self.parse_joint(0)
        self.line_number += 1

    def parse_joint(self, parent):
        self.joint_parents.append(parent)

        index = len(self.joint_names)
        self.joint_names.append(self.lines[self.line_number].split()[1])
        self.line_number += 2

        if self.lines[self.line_number].startswith('OFFSET'):
            self.joint_offsets.append(self.parse_offset(self.lines[self.line_number]))
        else:
            print('cannot find joint offset')
        self.line_number += 1

        if self.lines[self.line_number].startswith('CHANNELS'):
            channels = self.parse_channels(self.lines[self.line_number])
            if self.lines[self.line_number].split()[1] == '3':
                self.joint_rotation_channels.append((channels[0], channels[1], channels[2]))
        else:
            print('cannot find joint channels')
        self.line_number += 1

        while self.lines[self.line_number].startswith('JOINT') or \
                self.lines[self.line_number].startswith('End'):
            if self.lines[self.line_number].startswith('JOINT'):
                self.parse_joint(index)
            elif self.lines[self.line_number].startswith('End'):
                self.parse_end(index)
        self.line_number += 1

    def parse_end(self, parent):
        self.joint_parents.append(parent)

        self.joint_names.append(self.joint_names[parent] + '_end')
        self.line_number += 2

        if self.lines[self.line_number].startswith('OFFSET'):
            self.joint_offsets.append(self.parse_offset(self.lines[self.line_number]))
        else:
            print('cannot find joint offset')
        self.line_number += 2

    def analyze(self):
        if not self.lines[self.line_number].startswith('HIERARCHY'):
            print('cannot find hierarchy')
        self.line_number += 1

        if self.lines[self.line_number].startswith('ROOT'):
            self.parse_root()
    
        return self.joint_names, self.joint_parents, self.joint_offsets


def forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    m = len(joint_name)
    print(m)
    joint_positions = np.zeros((m, 3), dtype=np.float64)
    joint_rotation = np.zeros((m, 4), dtype=np.float64)
    channels = motion_data[frame_id]
    rotations = np.zeros((m, 3), dtype=np.float64)
    cnt = 1
    for i in range(m):
        if '_end' not in joint_name[i]:
            for j in range(3):
                rotations[i][j] = channels[cnt * 3 + j]
            cnt += 1
    for i in range(m):
        parent = joint_parent[i]
        if parent == -1:
            for j in range(3):
                joint_positions[0][j] = channels[j]
            joint_rotation[0] = R.from_euler('XYZ', [rotations[0][0], rotations[0][1],rotations[0][2]], degrees=True).as_quat()
        else:
            if '_end' in joint_name[i]:
                joint_rotation[i] = np.array([0, 0, 0, 1])
                joint_positions[i] = joint_positions[parent] + R.from_quat(joint_rotation[parent]).as_matrix() @ joint_offset[i]
            else:
                rotation = R.from_euler('XYZ', [rotations[i][0], rotations[i][1],rotations[i][2]], degrees=True)
                joint_rotation[i] = (R.from_quat(joint_rotation[parent]) * rotation).as_quat()
                joint_positions[i] = joint_positions[parent] + R.from_quat(joint_rotation[parent]).as_matrix() @ joint_offset[i]
    
    # 重新排列joint_orientations数组的顺序为xyzw
    joint_rotation = joint_rotation[:, [0,1,2,3]]
        
    return joint_positions, joint_rotation


def get_transform(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    m = len(joint_name)
    print(m)
    # joint_positions = np.zeros((m, 3), dtype=np.float64)
    joint_positions = np.array(joint_offset,dtype=np.float64)
    joint_positions[:,1]*=-1

    joint_rotation = np.zeros((m, 4), dtype=np.float64)
    channels = motion_data[frame_id]
    rotations = np.zeros((m, 3), dtype=np.float64)
    # 不变化的 joint_offset 
    cnt = 1
    for i in range(m):
        if '_end' not in joint_name[i]:
            for j in range(3):
                rotations[i][j] = channels[cnt*3+j]
            cnt+=1
    # 看bvh了
    rotations[:, 0] *= -1  # 第一列乘以-1
    rotations[:, 2] *= -1  # 第三列乘以-1 
    print(len(rotations))
    # 所有的角度变换成弧度   
    # z y x
    for i in range(len(rotations)):
        for j in range(len(rotations[i])):
            rotations[i][j] = np.radians(rotations[i][j])
    for i in range(len(rotations)):
        # 创建绕Z轴旋转的四元数
        RotationZ = R.from_rotvec([0, 0, rotations[i][0]])
        # 创建绕Y轴旋转的四元数
        RotationY = R.from_rotvec([0, rotations[i][1], 0])
        # 创建绕X轴旋转的四元数
        RotationX = R.from_rotvec([ rotations[i][2], 0, 0])
        try:
            # 计算组合的旋转四元数
            joint_rotation[i] = (RotationZ * RotationY * RotationX).as_quat()
        except ValueError as e:
            # 如果出现异常，将 joint_rotation[i] 设置为默认的四元数值
            print(f"Error in frame {frame_id}, setting default quaternion:", e)
            joint_rotation[i] = [0, 0, 0, 1]
        print(f"fdddddddddddddddd,{joint_rotation[i]}")
    return joint_positions,joint_rotation

def load_motion_data(bvh_file_path):
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


# def animation(joint_names, joint_parents, joint_offsets, motion_data):
#     frame_num = motion_data.shape[0]
    
#     class UpdateHandle:
#         def __init__(self):
#             self.current_frame = 0
#         def update_func(self):
#             joint_positions, joint_orientations = forward_kinematics(joint_names, \
#                 joint_parents, joint_offsets, motion_data, self.current_frame)
#             # 构造成json格式
#             result = {}
#             for i in range(len(joint_names)-1):
#                 result[joint_names[i]]={
#                     "positon":joint_positions[i].tolist(),
#                     "rotaion":joint_orientations[i].tolist()
#                 }
#             json_str = json.dumps(result)
           
#             # 将 JSON 数据转换为字符串

#             print(joint_names)
#             print(joint_positions)
#             print(joint_orientations)
#             print(self.current_frame)
#             print(len(joint_names))
#             print(len(joint_positions))
#             print(len(joint_orientations))
#             print(result)
#             # viewer.show_pose(joint_names, joint_positions, joint_orientations)
#             self.current_frame = (self.current_frame + 1) % frame_num
            

#     handle = UpdateHandle()
#     while True:
#         handle.update_func()
#         time.sleep(3)

# 名字改写功能

NAME_SELF = {
    'Hips': "pelvis", 
    # 'Spine':, 
    # 'Spine1', 
    # 'Spine2',
    'Spine3':"spine", 
    'Neck':"neck", 
    # 'Neck1', 
    'Head':"head", 
    # 'HeadEnd', 
    # 'HeadEnd_end', 
    'RightShoulder': "shoulder_r",
    'RightArm':"upperarm_r", 
    'RightForeArm':"lowerarm_r",
    'RightHand':"hand_r",
    # 'RightHandMiddle1':"middle1_r",
    'RightHandMiddle2':"middle1_r", 
    'RightHandMiddle3':"middle2_r",
    'RightHandMiddle4':"middle3_r",
    # 'RightHandMiddle4_end', 
    # 'RightHandRing':"ring1_r", 
    # 'RightHandRing1':"ring2_r", 
    'RightHandRing2':"ring1_r", 
    'RightHandRing3':"ring2_r", 
    'RightHandRing4':"ring3_r", 
    # 'RightHandRing4_end', 
    # 'RightHandPinky':"little1_r", 
    # 'RightHandPinky1':"little2_r",
    'RightHandPinky2':"little1_r",
    'RightHandPinky3':"little2_r", 
    'RightHandPinky4':"little3_r",
    # 'RightHandPinky4_end', 
    # 'RightHandIndex':"index1_r",
    # 'RightHandIndex1':"index2_r", 
    'RightHandIndex2':"index1_r",
    'RightHandIndex3':"index2_r", 
    'RightHandIndex4':"index3_r", 
    # 'RightHandIndex4_end',
    # 'RightHandThumb1':"thumb1_r",
    'RightHandThumb2':"thumb1_r", 
    'RightHandThumb3':"thumb2_r", 
    'RightHandThumb4':"thumb3_r", 
    # 'RightHandThumb4_end', 
    'LeftShoulder':"shoulder_l", 
    'LeftArm':"upperarm_l", 
    'LeftForeArm':"lowerarm_l",
    'LeftHand':"hand_l", 
    # 'LeftHandMiddle1':"middle1_l", 
    'LeftHandMiddle2':"middle1_l", 
    'LeftHandMiddle3':"middle2_l", 
    'LeftHandMiddle4':"middle3_l", 
    # 'LeftHandMiddle4_end', 
    # 'LeftHandRing':"ring1_l", 
    # 'LeftHandRing1':"ring2_l", 
    'LeftHandRing2':"ring1_l", 
    'LeftHandRing3':"ring2_l",
    'LeftHandRing4':"ring3_l", 
    # 'LeftHandRing4_end', 
    # 'LeftHandPinky':"little1_l", 
    # 'LeftHandPinky1':"little2_l", 
    'LeftHandPinky2':"little1_l", 
    'LeftHandPinky3':"little2_l", 
    'LeftHandPinky4':"little3_l", 
    # 'LeftHandPinky4_end', 
    # 'LeftHandIndex':"index1_l", 
    # 'LeftHandIndex1':"index2_l", 
    'LeftHandIndex2':"index1_l", 
    'LeftHandIndex3':"index2_l", 
    'LeftHandIndex4':"index3_l", 
    # 'LeftHandIndex4_end', 
    # 'LeftHandThumb1':"thumb1_l", 
    'LeftHandThumb2':"thumb1_l", 
    'LeftHandThumb3':"thumb2_l", 
    'LeftHandThumb4':"thumb3_l",
    # 'LeftHandThumb4_end', 
    'RightUpLeg':"thigh_r", 
    'RightLeg':"calf_r", 
    'RightFoot':"foot_r",
    'RightForeFoot':"toes_r", 
    # 'RightToeBase', 
    # 'RightToeBaseEnd', 
    # 'RightToeBaseEnd_end',
    'LeftUpLeg':"thigh_l", 
    'LeftLeg':"calf_l", 
    'LeftFoot':"foot_l", 
    'LeftForeFoot':"toes_l", 
    # 'LeftToeBase', 
    # 'LeftToeBaseEnd', 
    # 'LeftToeBaseEnd_end'
}
NAME_DOLLARS = {
    'Hips':"pelvis", 
    'Spine':"spine", 
    'Chest':"chest",
    'Neck':"neck", 
    'Head':"head", 
    # 'Head_end':"head",
    'LeftShoulder':"shoulder_l", 
    'LeftUpperArm':"upperarm_l",
    'LeftLowerArm':"lowerarm_l", 
    'LeftHand':"hand_l",
    'LeftIndexProximal':"index1_l",
    'LeftIndexIntermediate':"index2_l", 
    'LeftIndexDistal':"index3_l",
    # 'LeftIndexDistal_end':"index3_l",
    'LeftMiddleProximal':"middle1_l", 
    'LeftMiddleIntermediate':"middle2_l",
    'LeftMiddleDistal':"middel3_l", 
    # 'LeftMiddleDistal_end':"middel3_l",
    'LeftLittleProximal':"little1_l", 
    'LeftLittleIntermediate':"little2_l",
    'LeftLittleDistal':"little3_l",
    # 'LeftLittleDistal_end':"little3_l",
    'LeftRingProximal':"ring1_l", 
    'LeftRingIntermediate':"ring2_l", 
    'LeftRingDistal':"ring3_l", 
    # 'LeftRingDistal_end':"ring3_l", 
    'LeftThumbProximal':"thumb1_l", 
    'LeftThumbIntermediate':"thumb2_l", 
    'LeftThumbDistal':"thumb3_l", 
    # 'LeftThumbDistal_end':"thumb3_l", 
    'RightShoulder':"shoulder_r",
    'RightUpperArm':"upperarm_r", 
    'RightLowerArm':"lowerarm_r", 
    'RightHand':"hand_r", 
    'RightIndexProximal':"index1_r", 
    'RightIndexIntermediate':"index2_r", 
    'RightIndexDistal':"index3_r", 
    # 'RightIndexDistal_end':"index3_r", 
    'RightMiddleProximal':"middle1_r", 
    'RightMiddleIntermediate':"middle2_r",
    'RightMiddleDistal':"middle3_r", 
    # 'RightMiddleDistal_end':"middle3_r",
    'RightLittleProximal':"little1_r", 
    'RightLittleIntermediate':"little2_r", 
    'RightLittleDistal':"littel3_r", 
    # 'RightLittleDistal_end':"littel3_r", 
    'RightRingProximal':"ring1_r", 
    'RightRingIntermediate':"ring2_r", 
    'RightRingDistal':"ring3_r",
    # 'RightRingDistal_end':"ring3_r",
    'RightThumbProximal':"thumb1_r", 
    'RightThumbIntermediate':"thumb2_r",
    'RightThumbDistal':"thumb3_r", 
    # 'RightThumbDistal_end':"thumb3_r",
    'LeftUpperLeg':"thigh_l",
    'LeftLowerLeg':"calf_l",
    'LeftFoot':"foot_l",
    'LeftToes':"toes_l",
    # 'LeftToes_end',
    'RightUpperLeg':"thigh_r", 
    'RightLowerLeg':"calf_r",
    'RightFoot':"foot_r", 
    'RightToes':"toes_r", 
    # 'RightToes_end'
}

NEW_SCOTT_NAME = {
    'pelvis':"pelvis",
    'left_hip':"thigh_l", 
    'left_knee':"calf_l", 
    'left_ankle':"foot_l", 
    'left_foot':"toes_l", 
    # 'left_foot_end':"", 
    'right_hip':"thigh_r", 
    'right_knee':"calf_r", 
    'right_ankle':"foot_r", 
    'right_foot':"ball_l", 
    # 'right_foot_end', 
    'spine1':"spine", 
    # 'spine2', 
    # 'spine3', 
    'neck':"neck", 
    'head':"head", 
    # 'jaw':"", 
    # 'jaw_end', 
    # 'left_eye_smplhf', 
    # 'left_eye_smplhf_end', 
    # 'right_eye_smplhf', 
    # 'right_eye_smplhf_end', 
    'left_collar':"shoulder_l", 
    'left_shoulder':"upperarm_l", 
    'left_elbow':"lowerarm_l", 
    'left_wrist':"hand_l", 
    'left_index1':"index1_l", 
    'left_index2':"indexl_l", 
    'left_index3':"index3_l", 
    # 'left_index3_end', 
    'left_middle1':'middle1_l', 
    'left_middle2':"middle2_l",
    'left_middle3':"middle3_l",
    # 'left_middle3_end', 
    'left_pinky1':"little1_l", 
    'left_pinky2':"little2_l",
    'left_pinky3':'little3_l', 
    # 'left_pinky3_end', 
    'left_ring1':'ring1_l',
    'left_ring2':"ring2_l",
    'left_ring3':'ring3_l', 
    # 'left_ring3_end', 
    'left_thumb1':'thumb1_l',
    'left_thumb2':"thumb2_l",
    'left_thumb3':'thumb3_l', 
    # 'left_thumb3_end',
    'right_collar':"shoulder_r", 
    'right_shoulder':'upperarm_r', 
    'right_elbow':"lowerarm_r", 
    'right_wrist':"hand_r", 
    'right_index1':"index1_r", 
    'right_index2':'index2_r', 
    'right_index3':"index3_r", 
    # 'right_index3_end':"", 
    'right_middle1':"middle1_r", 
    'right_middle2':"middle2_r",
    'right_middle3':"middle3_r", 
    # 'right_middle3_end',
    'right_pinky1':"little1_r", 
    'right_pinky2':"little2_r", 
    'right_pinky3':"little3_r",
    # 'right_pinky3_end', 
    'right_ring1':'ring1_r', 
    'right_ring2':"ring2_r", 
    'right_ring3':"ring3_r", 
    # 'right_ring3_end',
    'right_thumb1':"thumb1_r",
    'right_thumb2':"thumb2_r", 
    'right_thumb3':"thumb3_r", 
    # 'right_thumb3_end'
}


def rename(origin_name):
    new_name = NEW_SCOTT_NAME.get(origin_name,"")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    return new_name if new_name else origin_name




def animation(joint_names, joint_parents, joint_offsets, motion_data):
    frame_num = motion_data.shape[0]
    class UpdateHandle:
        def __init__(self):
            self.current_frame = 0

        def update_func(self):
            joint_positions, joint_orientations = get_transform(joint_names,
                joint_parents, joint_offsets, motion_data, self.current_frame)
            # 构造成json格式
            result = {}
            for i in range(len(joint_names)-1):
                # 将位置信息和旋转信息分开
                # result[joint_names[i]] = {
                #     "position": joint_positions[i].tolist() ,
                #     "rotation": joint_orientations[i].tolist()
                # }
                new_name = rename(joint_names[i])
                # if new_name:
                #     print(new_name)
                if new_name:
                    result[new_name] = joint_positions[i].tolist() + joint_orientations[i].tolist()
                continue

                # result[rename(joint_names[i])] = joint_positions[i].tolist() + joint_orientations[i].tolist()

                
                
            json_str = json.dumps(result)
            return json_str
        def increment_frame(self):
            self.current_frame = (self.current_frame + 1) % frame_num
            
        def reset_current_frame(self):
            self.current_frame = 0
    handle = UpdateHandle()
    return handle


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class WebSocketServer:
    def __init__(self, host, port, joint_names, joint_parents, joint_offsets, motion_data):
        self.host = host
        self.port = port
        self.animation_handle = animation(joint_names, joint_parents, joint_offsets, motion_data)
        self.isFirst = False

    async def start_server(self):
        start_server = websockets.serve(self.server_handler, self.host, self.port)
        await start_server

    async def server_handler(self, websocket, path):
        try:
            message = await websocket.recv()
            print(f"Received message: {message}")
            while True:
                if self.isFirst:
                    message = await websocket.recv()
                    print(f"Received current: {message}")
                # 调用 animation 函数得到 JSON 字符串
                json_str = self.animation_handle.update_func()
                # 发送 JSON 字符串到客户端
                await websocket.send(json_str)
                # 更新帧
                self.animation_handle.increment_frame()
                time.sleep(0.02)
                self.isFirst = True
        except websockets.exceptions.ConnectionClosedError:
            print("Client disconnected")
            self.isFirst = False
        except websockets.exceptions.ConnectionClosedOK:
            print("主动关闭")
            self.isFirst = False
            self.animation_handle.reset_current_frame()
            
            
            

# def main():
#     bvh_file_path = '1_wayne_0_1_8.bvh'
#     parser = HierarchyParser(bvh_file_path)
#     joint_names, joint_parents, joint_offsets = parser.analyze() 
#     motion_data = load_motion_data(bvh_file_path)
#     print(joint_names)
#     print(joint_parents)
#     # 每一个的偏移量
#     print(joint_offsets)
    
    
#     print(motion_data)
#     print(len(motion_data))
#     print(len(motion_data[0]))
    
#     animation(joint_names, joint_parents, joint_offsets, motion_data)


def main():
    # 这里是你的动画数据
    bvh_file_path = 'res_2_scott_0_4_4.bvh'
    parser = HierarchyParser(bvh_file_path)
    joint_names, joint_parents, joint_offsets = parser.analyze() 
    # print(joint_names)
    print(joint_offsets)
    print(len(joint_offsets))
    # print(joint_parents)
    motion_data = load_motion_data(bvh_file_path)
    # print(motion_data)
    
    server = WebSocketServer("localhost", 8888, joint_names, joint_parents, joint_offsets, motion_data)
    asyncio.get_event_loop().run_until_complete(server.start_server())
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    main()

# def main():
#     bvh_file_path = '1_wayne_0_1_8.bvh'
#     parser = HierarchyParser(bvh_file_path)
#     joint_names, joint_parents, joint_offsets = parser.analyze()
#     print(joint_names)
# main()