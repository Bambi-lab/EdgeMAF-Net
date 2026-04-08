import warnings
warnings.filterwarnings('ignore')
from edgemaf import EdgeMAF

if __name__ == '__main__':
    model = EdgeMAF('ultralytics/cfg/models/EdgeMAF-Net.yaml')
    model.train(data='dataset',
                cache=False,
                imgsz=224,
                epochs=200,
                batch=64,
                close_mosaic=0,
                workers=0,
                patience=50,
                optimizer='SGD', 
                project='runs/EdgeMAF-Net',
                name='test',
                pretrained=False,
                )