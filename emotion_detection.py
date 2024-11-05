# Importa el método detect desde main.py
from main import detect
import torch
# Especifica la ruta del video
video_path = r"C:\Users\USER\Videos\pruebas\x.mp4"

# Configura los parámetros opcionales si es necesario
img_size = 512
conf_thres = 0.5
iou_thres = 0.45
device = ''
augment = False
agnostic_nms = False

# Llama al método detect con los parámetros necesarios
with torch.no_grad():
    emotion_dict = detect(video_path=video_path, img_size=img_size, conf_thres=conf_thres, iou_thres=iou_thres, device=device, augment=augment, agnostic_nms=agnostic_nms)

# Imprime el diccionario de emociones
print(emotion_dict)
