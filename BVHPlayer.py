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
    joint_positions = np.zeros((m, 3), dtype=np.float64)
    joint_orientations = np.zeros((m, 4), dtype=np.float64)
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
            joint_orientations[0] = R.from_euler('XYZ', [rotations[0][0], rotations[0][1], rotations[0][2]], degrees=True).as_quat()
        else:
            if '_end' in joint_name[i]:
                joint_orientations[i] = np.array([0, 0, 0, 1])
                joint_positions[i] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).as_matrix() @ joint_offset[i]
            else:
                rotation = R.from_euler('XYZ', [rotations[i][0], rotations[i][1], rotations[i][2]], degrees=True)
                joint_orientations[i] = (R.from_quat(joint_orientations[parent]) * rotation).as_quat()
                joint_positions[i] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).as_matrix() @ joint_offset[i]
    return joint_positions, joint_orientations


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

NAME = {
    'Hips': "pelvis", 
    # 'Spine':, 
    # 'Spine1', 
    # 'Spine2',
    'Spine3':"spine", 
    # 'Neck', 
    # 'Neck1', 
    # 'Head', 
    # 'HeadEnd', 
    # 'HeadEnd_end', 
    # 'RightShoulder',
    # 'RightArm', 
    # 'RightForeArm',
    # 'RightHand',
    # 'RightHandMiddle1',
    # 'RightHandMiddle2', 
    # 'RightHandMiddle3',
    # 'RightHandMiddle4',
    # 'RightHandMiddle4_end', 
    # 'RightHandRing', 
    # 'RightHandRing1', 
    # 'RightHandRing2', 
    # 'RightHandRing3', 
    # 'RightHandRing4', 
    # 'RightHandRing4_end', 
    # 'RightHandPinky', 
    # 'RightHandPinky1',
    # 'RightHandPinky2',
    # 'RightHandPinky3', 
    # 'RightHandPinky4',
    # 'RightHandPinky4_end', 
    # 'RightHandIndex',
    # 'RightHandIndex1', 
    # 'RightHandIndex2',
    # 'RightHandIndex3', 
    # 'RightHandIndex4', 
    # 'RightHandIndex4_end',
    # 'RightHandThumb1',
    # 'RightHandThumb2', 
    # 'RightHandThumb3', 
    # 'RightHandThumb4', 
    # 'RightHandThumb4_end', 
    # 'LeftShoulder', 
    # 'LeftArm', 
    # 'LeftForeArm',
    # 'LeftHand', 
    # 'LeftHandMiddle1', 
    # 'LeftHandMiddle2', 
    # 'LeftHandMiddle3', 
    # 'LeftHandMiddle4', 
    # 'LeftHandMiddle4_end', 
    # 'LeftHandRing', 
    # 'LeftHandRing1', 
    # 'LeftHandRing2', 
    # 'LeftHandRing3',
    # 'LeftHandRing4', 
    # 'LeftHandRing4_end', 
    # 'LeftHandPinky', 
    # 'LeftHandPinky1', 
    # 'LeftHandPinky2', 
    # 'LeftHandPinky3', 
    # 'LeftHandPinky4', 
    # 'LeftHandPinky4_end', 
    # 'LeftHandIndex', 
    # 'LeftHandIndex1', 
    # 'LeftHandIndex2', 
    # 'LeftHandIndex3', 
    # 'LeftHandIndex4', 
    # 'LeftHandIndex4_end', 
    # 'LeftHandThumb1', 
    # 'LeftHandThumb2', 
    # 'LeftHandThumb3', 
    # 'LeftHandThumb4',
    # 'LeftHandThumb4_end', 
    # 'RightUpLeg', 
    # 'RightLeg', 
    # 'RightFoot',
    # 'RightForeFoot', 
    # 'RightToeBase', 
    # 'RightToeBaseEnd', 
    # 'RightToeBaseEnd_end',
    # 'LeftUpLeg', 
    # 'LeftLeg', 
    # 'LeftFoot', 
    # 'LeftForeFoot', 
    # 'LeftToeBase', 
    # 'LeftToeBaseEnd', 
    # 'LeftToeBaseEnd_end'
}



def rename(origin_name):
    new_name = NAME[origin_name]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    return new_name if new_name else origin_name




def animation(joint_names, joint_parents, joint_offsets, motion_data):
    frame_num = motion_data.shape[0]
    class UpdateHandle:
        def __init__(self):
            self.current_frame = 0

        def update_func(self):
            joint_positions, joint_orientations = forward_kinematics(joint_names,
                joint_parents, joint_offsets, motion_data, self.current_frame)
            # 构造成json格式
            result = {}
            for i in range(len(joint_names)-1):
                result[joint_names[i]] = {
                    "position": joint_positions[i].tolist(),
                    "rotation": joint_orientations[i].tolist()
                }
            json_str = json.dumps(result)
            return json_str
        def increment_frame(self):
            self.current_frame = (self.current_frame + 1) % frame_num
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

    async def start_server(self):
        start_server = websockets.serve(self.server_handler, self.host, self.port)
        await start_server

    async def server_handler(self, websocket, path):
        try:
            message = await websocket.recv()
            print(f"Received message: {message}")
            while True:
                # 调用 animation 函数得到 JSON 字符串
                json_str = self.animation_handle.update_func()
                # 发送 JSON 字符串到客户端
                await websocket.send(json_str)
                # 更新帧
                self.animation_handle.increment_frame()
                time.sleep(3)
        except websockets.exceptions.ConnectionClosedError:
            print("Client disconnected")
            

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

# def main():
#     # 这里是你的动画数据
#     bvh_file_path = '1_wayne_0_1_8.bvh'
#     parser = HierarchyParser(bvh_file_path)
#     joint_names, joint_parents, joint_offsets = parser.analyze() 
#     motion_data = load_motion_data(bvh_file_path)
    
#     server = WebSocketServer("localhost", 8888, joint_names, joint_parents, joint_offsets, motion_data)
#     asyncio.get_event_loop().run_until_complete(server.start_server())
#     asyncio.get_event_loop().run_forever()

# if __name__ == "__main__":
#     main()

def main():
    bvh_file_path = '1_wayne_0_1_8.bvh'
    parser = HierarchyParser(bvh_file_path)
    joint_names, joint_parents, joint_offsets = parser.analyze()
    print(joint_names)
main()