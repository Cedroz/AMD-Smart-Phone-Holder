"""CPU MediaPipe worker; never imports robot or serial modules."""
import argparse
import base64
from importlib.metadata import version
import os
import socket
import sys

from followme.environment import verified_model
from followme.protocol import JsonSocket, VERSION


def validate_frame(message):
    if message.get('type') != 'frame' or message.get('camera_role') not in ('external','wrist'):
        raise ValueError('Invalid frame type or camera role')
    for key in ('sequence','capture_monotonic_ns'):
        if type(message.get(key)) is not int or message[key] < 0:
            raise ValueError(f'Invalid {key}')
    if not isinstance(message.get('request_id'),str) or not message['request_id']:
        raise ValueError('Missing request ID')
    return base64.b64decode(message['jpeg_base64'],validate=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fd',type=int,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--manifest',required=True)
    args=parser.parse_args()
    channel=JsonSocket(socket.socket(fileno=args.fd))
    try:
        manifest=verified_model(args.model,args.manifest)
        os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(1920*1080)
        import cv2
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        options=vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.model,delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,num_faces=2,
            output_facial_transformation_matrixes=True)
        # Independent stateful VIDEO instances avoid mixing external/wrist histories.
        with vision.FaceLandmarker.create_from_options(options) as external, vision.FaceLandmarker.create_from_options(options) as wrist:
            models={'external':external,'wrist':wrist}
            timestamps={'external':-1,'wrist':-1}
            channel.send({'type':'ready','protocol_version':VERSION,'interpreter':sys.executable,
                          'mediapipe_version':version('mediapipe'),'model_sha256':manifest['sha256']})
            while True:
                try:
                    message=channel.receive(timeout=3600)
                except EOFError:
                    break
                if message.get('type')=='shutdown':
                    break
                identity={key:message.get(key) for key in ('request_id','camera_role','sequence','capture_monotonic_ns')}
                try:
                    raw=validate_frame(message)
                    if not raw.startswith(b'\xff\xd8'):
                        raise ValueError('Expected a JPEG image')
                    # OpenCV pixel limit is set before import; also enforce dimensions.
                    frame=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
                    if frame is None or frame.shape[0]>1080 or frame.shape[1]>1920:
                        raise ValueError('Invalid JPEG or image dimensions')
                    role=message['camera_role']
                    ms=message['capture_monotonic_ns']//1_000_000
                    if ms<=timestamps[role]:
                        raise ValueError('Frame timestamps must increase for each camera')
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    result=models[role].detect_for_video(image,ms)
                    timestamps[role]=ms
                    channel.send({'type':'result',**identity,'face_count':len(result.face_landmarks),
                        'landmarks':[[{'x':p.x,'y':p.y,'z':p.z} for p in face] for face in result.face_landmarks],
                        'transformation_matrices':[m.tolist() for m in result.facial_transformation_matrixes],
                        'error':None})
                except Exception as exc:
                    channel.send({'type':'result',**identity,'error':f'{type(exc).__name__}: {exc}'})
    finally:
        channel.close()


if __name__=='__main__':
    main()
