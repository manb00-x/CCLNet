from ultralytics import YOLOMM
import warnings
import torch
from ultralytics import YOLO

warnings.filterwarnings('ignore', category=UserWarning)



if __name__ == '__main__':



    model = YOLOMM('CCLNet.yaml')
    model.train(
        data='RGBD-ID.yaml',
        epochs=300,
        batch=8,
        workers=4,
        imgsz=640,
        exist_ok=True,
        patience=50,
    )
