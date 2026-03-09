import os
import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# 1. Configuración del modelo (ResNet50)
# Cargamos el modelo y eliminamos la última capa (clasificación)
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
modules = list(resnet.children())[:-1]  # Quitamos la capa 'fc'
feature_extractor = torch.nn.Sequential(*modules)
feature_extractor.eval()

# 2. Preprocesamiento de imágenes (Requerido por ResNet)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Procesamiento de las carpetas
base_path = './data'
categories = ['metal', 'paper', 'glass']
data_records = []

with torch.no_grad():
    for category in categories:
        folder_path = os.path.join(base_path, category)
        if not os.path.exists(folder_path):
            continue
            
        print(f"Procesando categoría: {category}")
        for img_name in tqdm(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            try:
                # Cargar y transformar imagen
                img = Image.open(img_path).convert('RGB')
                input_tensor = preprocess(img).unsqueeze(0)
                
                # Extraer features (vector de 2048 dimensiones para ResNet50)
                features = feature_extractor(input_tensor).flatten().numpy()
                
                # Guardar en diccionario
                record = {'label': category, 'image_name': img_name}
                for i, f in enumerate(features):
                    record[f'feat_{i}'] = f
                
                data_records.append(record)
            except Exception as e:
                print(f"Error en {img_name}: {e}")

# 4. Crear el DataFrame final
df = pd.DataFrame(data_records)
print(f"\nDataFrame creado con éxito. Forma: {df.shape}")

df.head(10)