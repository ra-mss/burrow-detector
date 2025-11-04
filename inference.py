import os
import rasterio
import torch
import numpy as np
from rasterio.windows import Window
from ultralytics import YOLO
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box
import config # ¡Importamos nuestra configuración!

def inference_on_large_tif(model_path, image_path, output_path):
    print("--- Iniciando Fase 4: Inferencia en Imagen Gigante ---")
    
    # 0. Cargar el modelo
    print(f"Cargando modelo desde {model_path}...")
    # (Tu log de Colab mostró que se cargó en CPU, forzamos CUDA si está disponible)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path).to(device)
    model.fuse()
    print(f"Modelo cargado en {device}.")

    # 1. Abrir la imagen gigante
    with rasterio.open(image_path) as src_tif:
        ancho_img = src_tif.width
        alto_img = src_tif.height
        img_crs = src_tif.crs
        img_transform = src_tif.transform

        # 2. Preparar la iteración
        coords_x = range(0, ancho_img, config.STRIDE)
        coords_y = range(0, alto_img, config.STRIDE)

        all_boxes_global_px = []
        all_scores = []
        all_class_ids = []

        print("Iniciando inferencia con ventana deslizante...")
        pbar = tqdm(total=len(coords_x) * len(coords_y), desc="Procesando parches")

        for y in coords_y:
            for x in coords_x:
                pbar.update(1)
                
                ventana = Window(x, y, config.PATCH_SIZE, config.PATCH_SIZE)
                ventana = ventana.intersection(Window(0, 0, ancho_img, alto_img))
                
                patch_array = src_tif.read(window=ventana)
                
                h, w = patch_array.shape[1], patch_array.shape[2]
                if h < config.PATCH_SIZE or w < config.PATCH_SIZE:
                    temp_array = np.zeros((patch_array.shape[0], config.PATCH_SIZE, config.PATCH_SIZE), dtype=patch_array.dtype)
                    temp_array[:, :h, :w] = patch_array
                    patch_array = temp_array
                
                patch_rgb = np.moveaxis(patch_array, 0, -1)
                
                if patch_rgb.shape[2] == 1:
                    patch_rgb = np.stack((patch_rgb[:,:,0],)*3, axis=-1)
                elif patch_rgb.shape[2] > 3:
                    patch_rgb = patch_rgb[:, :, :3]
                
                if patch_rgb.shape[2] != 3:
                    continue

                results = model(patch_rgb, conf=config.CONF_THRESHOLD, verbose=False)
                
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    bbox_local = boxes.xyxyn[i].cpu().numpy()
                    score = boxes.conf[i].cpu().item()
                    cls_id = boxes.cls[i].cpu().item()

                    l_xmin = bbox_local[0] * config.PATCH_SIZE
                    l_ymin = bbox_local[1] * config.PATCH_SIZE
                    l_xmax = bbox_local[2] * config.PATCH_SIZE
                    l_ymax = bbox_local[3] * config.PATCH_SIZE

                    g_xmin = l_xmin + x
                    g_ymin = l_ymin + y
                    g_xmax = l_xmax + x
                    g_ymax = l_ymax + y

                    all_boxes_global_px.append([g_xmin, g_ymin, g_xmax, g_ymax])
                    all_scores.append(score)
                    all_class_ids.append(cls_id)
        
        pbar.close()
        print(f"Detección preliminar completa. Se encontraron {len(all_boxes_global_px)} cajas (con duplicados).")

        # 8. Aplicar NMS
        if not all_boxes_global_px:
            print("No se encontraron detecciones.")
            return

        print("Aplicando NMS para fusionar duplicados...")
        boxes_tensor = torch.tensor(all_boxes_global_px, dtype=torch.float32).to(device)
        scores_tensor = torch.tensor(all_scores).to(device)
        
        try:
            from torchvision.ops import nms
            indices_finales = nms(boxes_tensor, scores_tensor, config.IOU_THRESHOLD)
            
            final_boxes = boxes_tensor[indices_finales]
            final_scores = scores_tensor[indices_finales]
            final_classes = torch.tensor(all_class_ids)[indices_finales]

        except ImportError:
            print("Error: torchvision no encontrado.")
            return

        conteo_final = len(final_boxes)
        print(f"\n--- ¡RESULTADO FINAL! ---")
        print(f"Conteo total de madrigueras (después de NMS): {conteo_final}")

        # 9. Guardar resultados como GeoPackage
        print(f"Guardando resultados en {output_path}...")
        geometries = []
        for bbox_px in final_boxes.cpu().numpy():
            g_xmin, g_ymin = rasterio.transform.xy(img_transform, bbox_px[1], bbox_px[0])
            g_xmax, g_ymax = rasterio.transform.xy(img_transform, bbox_px[3], bbox_px[2])
            geometries.append(box(min(g_xmin, g_xmax), min(g_ymin, g_ymax), max(g_xmin, g_xmax), max(g_ymin, g_ymax)))
            
        gdf_resultados = gpd.GeoDataFrame(
            {'clase': [model.names[int(c)] for c in final_classes.cpu().numpy()],
             'confianza': final_scores.cpu().numpy()},
            geometry=geometries,
            crs=img_crs
        )
        
        gdf_resultados.to_file(output_path, driver="GPKG")
        print("¡Archivo de resultados guardado!")
        print(f"Puedes abrir '{output_path}' en QGIS para ver las detecciones.")

if __name__ == "__main__":
    # ¡Asegúrate de que la ruta al modelo entrenado exista!
    # El script de Fase 3 lo copia a config.RUTA_MODELO_GUARDADO
    inference_on_large_tif(config.RUTA_MODELO_GUARDADO, 
                             config.RUTA_TIF_GIGANTE_INFERENCIA, 
                             config.RUTA_RESULTADOS_GPKG)