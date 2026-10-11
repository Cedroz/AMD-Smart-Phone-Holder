"""Isolated Phase 2 face/optional shoulder inference. No hardware access."""
import argparse
from contextlib import ExitStack
from importlib.metadata import version
import socket
import sys

from followme.environment import verified_model
from followme.protocol import JsonSocket, VERSION
from followme.vision_worker import validate_frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fd',type=int,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--threshold',type=float,default=.7)
    parser.add_argument('--pose-model')
    parser.add_argument('--pose-manifest')
    args=parser.parse_args()
    channel=JsonSocket(socket.socket(fileno=args.fd))
    try:
        manifest=verified_model(args.model,args.manifest)
        if not 0 < args.threshold <= 1: raise ValueError('Invalid threshold')
        import cv2
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        with ExitStack() as stack:
            options=vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=args.model,delegate=BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO,num_faces=2,output_facial_transformation_matrixes=True,
                min_face_detection_confidence=args.threshold,min_face_presence_confidence=args.threshold,min_tracking_confidence=args.threshold)
            models={role:stack.enter_context(vision.FaceLandmarker.create_from_options(options)) for role in ('external','wrist')}
            pose=None
            if args.pose_model:
                verified_model(args.pose_model,args.pose_manifest)
                pose=stack.enter_context(vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=args.pose_model,delegate=BaseOptions.Delegate.CPU),
                    running_mode=vision.RunningMode.VIDEO,num_poses=2,
                    min_pose_detection_confidence=.7,min_pose_presence_confidence=.7,min_tracking_confidence=.7)))
            timestamps={role:-1 for role in models}
            channel.send({'type':'ready','protocol_version':VERSION,'interpreter':sys.executable,
                          'mediapipe_version':version('mediapipe'),'model_sha256':manifest['sha256']})
            while True:
                try: message=channel.receive(3600)
                except EOFError: break
                if message.get('type')=='shutdown': break
                identity={key:message.get(key) for key in ('request_id','camera_role','sequence','capture_monotonic_ns')}
                try:
                    raw=validate_frame(message)
                    frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    if frame is None or frame.shape[0]>1080 or frame.shape[1]>1920: raise ValueError('Invalid JPEG dimensions')
                    role=message['camera_role']
                    stamp=message['capture_monotonic_ns']//1_000_000
                    if stamp<=timestamps[role]: raise ValueError('Timestamps must increase')
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    result=models[role].detect_for_video(image,stamp)
                    timestamps[role]=stamp
                    shoulders=None
                    if pose and role=='external' and len(result.face_landmarks)==1:
                        body=pose.detect_for_video(image,stamp)
                        if len(body.pose_landmarks)==1:
                            left,right=body.pose_landmarks[0][11:13]
                            nose=body.pose_landmarks[0][0]
                            face=result.face_landmarks[0]
                            face_x=(min(p.x for p in face)+max(p.x for p in face))/2
                            face_y=(min(p.y for p in face)+max(p.y for p in face))/2
                            associated=(nose.x-face_x)**2+(nose.y-face_y)**2<=.15**2
                            if associated and all(p.visibility is not None and p.visibility>=.7 and p.presence is not None and p.presence>=.7 for p in (left,right,nose)):
                                shoulders=[(left.x+right.x)/2,(left.y+right.y)/2]
                    channel.send({'type':'result',**identity,'face_count':len(result.face_landmarks),
                        'landmarks':[[{'x':p.x,'y':p.y,'z':p.z} for p in face] for face in result.face_landmarks],
                        'transformation_matrices':[m.tolist() for m in result.facial_transformation_matrixes],
                        'shoulder_center':shoulders,'error':None})
                except Exception as exc:
                    channel.send({'type':'result',**identity,'error':f'{type(exc).__name__}: {exc}'})
    finally:
        channel.close()


if __name__=='__main__': main()
