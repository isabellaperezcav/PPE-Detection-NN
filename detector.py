# Instala ultralytics:
#install ultralytics

# Verifica que se use tu tarjeta grafica:
#python
#import torch
#torch.cuda.is_available()

# Entrena tu red neuronal:

from ultralytics import YOLO
import torch
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    # Use the model
    results = model.train(data=r"C:\Users\ASUS\Desktop\Detector\custom.yaml", epochs=200, imgsz=640, batch=2, optimizer='Adam', patience=0,  lr0=1e-3, lrf=1e-6, )  # train the model
