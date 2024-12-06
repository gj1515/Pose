import os
import json
import numpy as np
import pickle


def parse_annot_per_image(data):
    annotations_per_image = {}
    for annotation in data['annotations']:
        if annotation['num_keypoints'] != 0 and not annotation['iscrowd']:
            if annotation['image_id'] not in annotations_per_image:
                annotations_per_image[annotation['image_id']] = []
            annotations_per_image[annotation['image_id']].append(annotation)

    images_info = {}
    for image_info in data['images']:
        images_info[image_info['id']] = image_info

    return annotations_per_image, images_info


def num_face_keypoints(coco_keypoints):
    # 0-4
    num_face = 0
    for i in range(5):
        if coco_keypoints[i * 3 + 2] > 0:
            num_face += 1
    return num_face


def num_body_keypoints(coco_keypoints, num_total):
    num_face = num_face_keypoints(coco_keypoints)
    return num_total - num_face


def parse_single_person_annot(annotations_per_image, images_info, MIN_BODY_KEYPOINTS, MIN_BODY_AREA):
    prepared_annotations = []

    for _, annotations in annotations_per_image.items():
        previous_centers = []
        valid_annotations = []
        #valid_area = []
        #valid_cent_dist = [] # dist to img center
        for annotation in annotations:
            num_body_pts = num_body_keypoints(annotation['keypoints'], annotation['num_keypoints'])

            if (num_body_pts < MIN_BODY_KEYPOINTS or annotation['area'] < MIN_BODY_AREA * MIN_BODY_AREA):
                continue

            person_center = [annotation['bbox'][0] + annotation['bbox'][2] / 2,
                             annotation['bbox'][1] + annotation['bbox'][3] / 2]
            is_close = False
            for previous_center in previous_centers:
                distance_to_previous = ((person_center[0] - previous_center[0]) ** 2
                                        + (person_center[1] - previous_center[1]) ** 2) ** 0.5
                if distance_to_previous < previous_center[2] * 0.3:
                    is_close = True
                    break
            if is_close:
                continue

            prepared_annotation = {
                'img_file': images_info[annotation['image_id']]['file_name'],
                'img_width': images_info[annotation['image_id']]['width'],
                'img_height': images_info[annotation['image_id']]['height'],
                'obj_pos': person_center,
                'obj_bbox': annotation['bbox']
            }

            keypoints = np.zeros((17, 3), dtype=int)
            # viz: 0: not labeled (in which case x=y=0, not in image)
            # 1: labeled but not visible, 2: labeled and visible
            w = prepared_annotation['img_width']
            h = prepared_annotation['img_height']
            for i in range(17):
                x = annotation['keypoints'][i * 3] # int
                y = annotation['keypoints'][i * 3 + 1]
                if x == 0 and y == 0:
                    continue

                if x < 0 or x >= w or y < 0 or y >= h:
                    continue # mark as invisible

                keypoints[i, 0] = x
                keypoints[i, 1] = y

                if annotation['keypoints'][i * 3 + 2] > 0:
                    keypoints[i, 2] = 1

            prepared_annotation['keypoints'] = keypoints

            valid_annotations.append(prepared_annotation)
            #valid_area.append(annotation['area'])

            # c_x = (person_center[0] - w / 2)
            # c_y = (person_center[1] - h / 2)
            # dist = c_x*c_x + c_y*c_y
            # valid_cent_dist.append(dist)

            previous_centers.append((person_center[0], person_center[1], annotation['bbox'][2], annotation['bbox'][3]))

        num_valid = len(valid_annotations)
        if num_valid == 1: # single person
            prepared_annotations.append(valid_annotations[0])
        # elif num_valid == 2: # two people
        #     #if valid_area[0] > valid_area[1]: # bigger (not good)
        #     if valid_cent_dist[0] < valid_cent_dist[1]: # close to center
        #         prepared_annotations.append(valid_annotations[0]) # only add main character
        #     else:
        #         prepared_annotations.append(valid_annotations[1]) # only add main character

    return prepared_annotations


def convert_keypoints(skeleton, prepared_annotations):
    # coco-17 to opose-18
    #reorder_map = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3] # 17
    #reorder_loc = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10] # 17

    for annotation in prepared_annotations:
        coco_key = annotation['keypoints']
        opose_key = np.zeros((skeleton.num_joints, 3), dtype='float32')

        for i in range(len(skeleton.coco_map)):
            if skeleton.coco_map[i] >= 0:
                opose_key[i] = coco_key[skeleton.coco_map[i]]

        # Add neck as a mean of shoulders
        if skeleton.coco_neck_idx >= 0:
            if coco_key[5,2] == 1 and coco_key[6,2] == 1:
                opose_key[skeleton.coco_neck_idx] = (coco_key[5] + coco_key[6]) / 2.

        annotation['keypoints'] = opose_key


def preprocess_single_person_coco(config, type='train'):
    if config.dataset == 'coco_eval':
        type = 'train'
    label_file = os.path.join('D:/Dev/Dataset/coco/annotations/' + type + '_label.pkl')
    if os.path.exists(label_file):
        print('Loading filtered annotations for {} from {}'.format(type, label_file))
        with open(label_file, 'rb') as f:
            return pickle.load(f)
    else:
        print('Filtering annotations for {}'.format(type))
        annot_name = os.path.join(config.data, 'annotations/person_keypoints_{}{}.json'.format(type, 2017))
        with open(annot_name, 'r') as f:
            data = json.load(f)

        annotations_per_image, images_info = parse_annot_per_image(data)

        MIN_BODY_KEYPOINTS = 7
        MIN_BODY_AREA = 80
        # Train: 29,634 (2 ppl) images --> 22,578 (1 person)
        # Val: 1,193 (2 ppl) images --> 937 (1 person)
        prepared_annotations = parse_single_person_annot(annotations_per_image, images_info, MIN_BODY_KEYPOINTS, MIN_BODY_AREA)

        # convert_keypoints(config.skeleton, prepared_annotations)

        print('Saving filtered annotations for {} to {}'.format(type, label_file))
        with open(label_file, 'wb') as f:
            pickle.dump(prepared_annotations, f)

        return prepared_annotations