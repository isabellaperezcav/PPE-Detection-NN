# Instala la herramienta de etiquetado:
#install labelme

# Si te pide actualizar pip:
#python.exe -m pip install --upgrade pip

# Ejecuta labelme:
#labelme

# Installa labelme2yolo para convertir Json a Yolo:
#pip install labelme2yolo

# Convierte tus archivos Json a Yolo:
#labelme2yolo --json_dir tu/direccion/sin/espacios

# Instala ultralytics:
#install ultralytics

# Verifica tu tarjeta grafica:
#python
#import torch
#torch.cuda.is_available()

# Entrena tu red neuronal:
#yolo task=segment mode=train epochs=30 data=C:\Users\ASUS\Desktop\Detector\custom.yaml model=yolov8m-seg.pt imgsz=640 batch=2


from ultralytics import YOLO
import torch
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    # Use the model
    results = model.train(data=r"C:\Users\ASUS\Desktop\Detector\custom.yaml", epochs=200, imgsz=640, batch=2, optimizer='Adam', patience=0,  lr0=1e-3, lrf=1e-6, )  # train the model