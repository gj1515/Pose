

'''
COCO-Keypoints
0 [ "nose",  -> 0
1 "left_eye", -> 15
2 "right_eye", -> 14
3 "left_ear", -> 17
4 "right_ear", -> 16
5 "left_shoulder", -> 5
6 "right_shoulder", -> 2
7 "left_elbow", -> 6
8 "right_elbow", -> 3
9 "left_wrist", -> 7
10 "right_wrist", -> 4
11 "left_hip", -> 11
12 "right_hip", -> 8
13 "left_knee", -> 12
14 "right_knee", -> 9
15 "left_ankle", -> 13
16 "right_ankle" ] -> 10
17 neck (virtually added)
'''



class Skeleton:
    def __init__(self, skel_type):
        if skel_type == 'opose14':
            '''
            Opose-14
            0 nose
            1 neck
            2 right_shoulder
            3 right_elbow
            4 right_wrist
            5 left_shoulder
            6 left_elbow
            7 left_wrist
            8 right_hip
            9 right_knee
            10 right_ankle
            11 left_hip
            12 left_knee
            13 left_ankle            
            '''
            self.names = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
                          'LShoulder', 'LElbow', 'LWrist',
                          'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle']
            self.parents = [1, -1, 1, 2, 3,
                            1, 5, 6,
                            1, 8, 9, 1, 11, 12]

            self.coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15]  # opose-14 from coco-17
            self.coco_neck_idx = 1  # <- avg of coco_lr_sho
            self.flipIndices = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13)]  # (r, l)
            # self.bone_colors = [(255, 0, 255),
            #                     (255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255),
            #                     (255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

            self.bone_colors = [(0, 165, 255),
                                (255, 0, 255), (255, 0, 255), (255, 0, 255), (0, 255, 0), (0, 255, 0), (0, 255, 0),
                                (255, 0, 255), (255, 0, 255), (255, 0, 255), (0, 255, 0), (0, 255, 0), (0, 255, 0)]

            self.rob22_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, -1, -1]
            self.rob23_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

            self.joint_color = (0, 0, 255)
            self.joint_thick = 3
            self.bone_thick = 5
            print('Skeleton Structure: OPOSE-14')

        elif skel_type == 'coco17':
            '''
            COCO-17
            0 nose
            1 left_eye
            2 right_eye
            3 left_ear
            4 right_ear
            5 left_shoulder
            6 right_shoulder
            7 left_elbow
            8 right_elbow
            9 left_wrist
            10 right_wrist
            11 left_hip
            12 right_hip
            13 left_knee
            14 right_knee
            15 left_ankle
            16 right_ankle
            '''
            self.names = ['Nose', 'LEye', 'REye', 'LEar', 'REar',
                          'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
                          'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']

            # Define parent joints (-1 is root)
            self.parents = [1, -1, -1, 1, 1,  # Nose to Ears
                            1, 1, 5, 6, 7, 8,  # Shoulders to Wrists
                            5, 6, 11, 12, 13, 14]  # Hips to Ankles

            # Define flip indices for left-right pairs
            self.flipIndices = [(1, 2),  # eyes
                                (3, 4),  # ears
                                (5, 6),  # shoulders
                                (7, 8),  # elbows
                                (9, 10),  # wrists
                                (11, 12),  # hips
                                (13, 14),  # knees
                                (15, 16)]  # ankles

            # Define bone colors (BGR format)
            self.bone_colors = [(0, 165, 255),  # orange for spine
                                (255, 0, 255), (255, 0, 255),  # magenta for face
                                (255, 0, 255), (255, 0, 255),
                                (255, 0, 255), (255, 0, 255),  # right side
                                (255, 0, 255), (255, 0, 255),
                                (0, 255, 0), (0, 255, 0),  # left side
                                (0, 255, 0), (0, 255, 0),
                                (0, 255, 0), (0, 255, 0),
                                (0, 255, 0), (0, 255, 0)]

            # Define mappings
            self.coco_map = list(range(17))  # identity mapping since this is COCO
            self.rob22_map = [0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]  # mapping to rob22
            self.rob23_map = [0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]  # mapping to rob23

            self.joint_color = (0, 0, 255)  # red color for joints
            self.joint_thick = 3
            self.bone_thick = 5
            print('Skeleton Structure: COCO-17')
        else:
            raise ValueError('Skeleton ' + skel_type + ' not available.')

        self.num_joints = len(self.names)
        self.bones = []
        self.bone_names = []
        for i, par in enumerate(self.parents):
            if par >= 0:
                self.bones.append((par, i)) # parJnt, curJnt
                self.bone_names.append('{}_{}'.format(self.names[par], self.names[i]))

        self.num_bones = len(self.bones)


