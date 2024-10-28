import os
root = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, root)
import cv2
import math
import json
import torch
import numpy as np
from collections import deque
from mmpose.apis.inferencers import MMPoseInferencer
import mmengine
from mmaction.apis import inference_skeleton, init_recognizer

def init_args(video=None):
    if video is None:
        video = './static/demo/clip_short.mp4'
    args = {
        'video': video,
        'config': 'models/config.py',
        'checkpoint': 'models/checkpoint.pth',
        'labelmap': 'models/labelmap.txt',
        'device': 'cuda:0',
        'step-size': 10,
        'extract_args': {
            'init_args': {
                'pose2d': 'rtmo', 
                'pose2d_weights': './checkpoint/rtmo_checkpoint.pth', 
                'scope': 'mmpose', 
                'device': 'cuda:0', 
                'det_model': None, 
                'det_weights': None, 
                'det_cat_ids': 0, 
                'pose3d': None, 
                'pose3d_weights': None, 
                'show_progress': False
            },
            'call_args': {
                'inputs': video, 
                'show': False, 
                'draw_bbox': True, 
                'draw_heatmap': False, 
                'bbox_thr': 0.5, 
                'nms_thr': 0.65, 
                'pose_based_nms': True, 
                'kpt_thr': 0.3, 
                'tracking_thr': 0.3, 
                'use_oks_tracking': False, 
                'disable_norm_pose_2d': False, 
                'disable_rebase_keypoint': False, 
                'num_instances': 1, 
                'radius': 3, 
                'thickness': 1, 
                'skeleton_style': 'openpose', 
                'black_background': False, 
                'vis_out_dir': '', 
                'pred_out_dir': '', 
                'vis-out-dir': './'
            }
        }
    }
    return args

def pre_processing(pose_data):
    # 모델에서 사용할 형태
    form = {
        'tid': None,
        'bbox_scores': [],
        'bboxes': [],
        'keypoints_visible': [],
        'keypoint_scores': [],
        'keypoints': []
    }
    # 전처리 과정
    for idx, pose in enumerate(pose_data):
        form['tid'] = idx
        form['bbox_scores'].append(pose['bbox_score'])
        form['bboxes'].append(pose['bbox'][0])
        form['keypoints_visible'].append(pose['keypoint_scores'])
        form['keypoint_scores'].append(pose['keypoint_scores'])
        form['keypoints'].append(pose['keypoints'])

    # 모델에서 사용할 형태에 전처리된 데이터 추가
    form['bbox_scores'] = np.array(form['bbox_scores'], dtype=np.float32)
    form['bboxes'] = np.array(form['bboxes'], dtype=np.float32)
    form['keypoints_visible'] = np.array(form['keypoints_visible'], dtype=np.float32)
    form['keypoint_scores'] = np.array(form['keypoint_scores'], dtype=np.float32)
    form['keypoints'] = np.array(form['keypoints'], dtype=np.float32)
    # 전처리된 데이터 반환
    return form

async def inference(video=None):
    yield f"data: {json.dumps({'message': f'Inference Start'})}\n\n"
    yield f"data: {json.dumps({'message': f'Initalizing Arguments...'})}\n\n"
    args = init_args(video)
    _init_args = args['extract_args']['init_args']
    _call_args = args['extract_args']['call_args']
    yield f"data: {json.dumps({'message': f'Initalizing Arguments... Done!'})}\n\n"

    yield f"data: {json.dumps({'message': f'Initalizing Extract Model...'})}\n\n"
    inferencer = MMPoseInferencer(**_init_args)
    yield f"data: {json.dumps({'message': f'Initalizing Extract Model... Done!'})}\n\n"
    yield f"data: {json.dumps({'message': f'Initalizing Selfharm Model...'})}\n\n"
    config = mmengine.Config.fromfile(args['config'])
    model = init_recognizer(config, args['checkpoint'], args['device'])
    labelmap = [x.strip() for x in open(args['labelmap']).readlines()]
    yield f"data: {json.dumps({'message': f'Initalizing Selfharm Model... Done!'})}\n\n"

    cap = cv2.VideoCapture(_call_args['inputs'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    frame_queue = deque(maxlen=args['step-size'])
    selfharm_result = []
    yield f"data: {json.dumps({'message': f'Extracting Video and Detecting Selfharm...'})}\n\n"
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                running_time = format(frame_index / fps, '.2f')
                skeletons = []
                progress = math.ceil(frame_index / frame_count * 100)
                
                temp_call_args = _call_args
                temp_call_args['inputs'] = frame
                results = inferencer(**temp_call_args)
                for result in results:
                    pred = result['predictions'][0]
                    pred.sort(key = lambda x: x['bbox'][0][0])
                    pose_data = pre_processing(pred)
                frame_queue.append(pose_data)
                if frame_index % args['step-size'] == 0 and len(frame_queue) == args['step-size']:
                    result = inference_skeleton(model, frame_queue, (frame_size))
                    # 행동 결과값 중 가장 높은 예측값을 가진 행동 가져오기
                    max_pred_index = result.pred_score.argmax().item()
                    # 숫자로된 행동 결과값을 행동 라벨 이름으로 매칭하기
                    action_label = labelmap[max_pred_index]
                    # 가장 높은 예측값을 가진 행동의 예측값 가저오기
                    confidence = result.pred_score[max_pred_index]
                    if (action_label != 'normal' and action_label != "hittingbody" and confidence > 0.95) or (action_label == 'normal' and confidence < 0.1):
                        if (action_label == 'choking_hand' or action_label == 'choking_cloth') and confidence > 0.95:
                            detail_label = 'choking'
                            action_label = 'selfharm'
                        elif (action_label == 'normal' and confidence < 0.1):
                            detail_label = 'undefined'
                            action_label = 'selfharm'
                        elif (confidence > 0.985):
                            detail_label = action_label
                            action_label = 'selfharm'
                        else:
                            detail_label = 'normal'
                            action_label = 'normal'
                    else:
                        detail_label = 'normal'
                        action_label = 'normal'
                    yield f"data: {json.dumps({'progress': f'{progress}'})}\n\n"
                    yield f"data: {json.dumps({'message': f'[{running_time}s] category: {action_label}, action: {detail_label}  confidence: {confidence}'})}\n\n"
                    if action_label == 'selfharm':
                        selfharm_result.append((running_time, detail_label, confidence))
                frame_index += 1
            else:
                yield f"data: {json.dumps({'progress': f'{100}'})}\n\n"
                break
    cap.release()
    yield f"data: {json.dumps({'message': f'Extracting Video and Detecting Selfharm... Done'})}\n\n"
    yield f"data: {json.dumps({'message': f'Inference Result: {selfharm_result}'})}\n\n"
    yield f"data: {json.dumps({'message': f'Visualize Result'})}\n\n"
    if len(selfharm_result) == 0:
        yield f"data: {json.dumps({'message': f'No Selfharm were detected.'})}\n\n"
    else:
        for row in selfharm_result:
            running_time, label, confidence = row
            yield f"data: {json.dumps({'message': f'[{running_time}s] {label}: {confidence}%'})}\n\n"
