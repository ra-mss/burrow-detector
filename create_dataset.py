import os
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from shapely.geometry import box
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import config  # ¡Importamos nuestra configuración!

def normalizar_para_yolo(bbox_pixel, patch_size_w, patch_size_h):
    """Convierte (xmin, ymin, xmax, ymax) a formato YOLO."""
    xmin, ymin, xmax, ymax = bbox_pixel
    w = xmax - xmin
    h = ymax - ymin
    center_x = xmin + (w / 2)
    center_y = ymin + (h / 2)
    norm_center_x = center_x / patch_size_w
    norm_center_y = center_y / patch_size_h
    norm_w = w / patch_size_w
    norm_h = h / patch_size_h
    return norm_center_x, norm_center_y, norm_w, norm_h

def main():
    print("--- Iniciando Fase 2: Creación de Dataset ---")
    print(f"Cargando imagen: {config.RUTA_TIF_RECORTE_ENTRENAR}")
    print(f"Cargando etiquetas: {config.RUTA_LABELS_GPKG}")

    dir_imagenes = os.path.join(config.RUTA_DATASET_PARCHES, "images")
    dir_etiquetas = os.path.join(config.RUTA_DATASET_PARCHES, "labels")
    os.makedirs(dir_imagenes, exist_ok=True)
    os.makedirs(dir_etiquetas, exist_ok=True)

    with rasterio.open(config.RUTA_TIF_RECORTE_ENTRENAR) as src_tif:
        labels_gdf = gpd.read_file(config.RUTA_LABELS_GPKG)

        if src_tif.crs != labels_gdf.crs:
            print("ADVERTENCIA: CRS no coincide. Reproyectando etiquetas...")
            labels_gdf = labels_gdf.to_crs(src_tif.crs)
            print("Reproyección completa.")

        labels_filtradas = labels_gdf[labels_gdf[config.CAMPO_CLASE_EN_GPKG] == config.CLASE_OBJETIVO]
        print(f"Se encontraron {len(labels_filtradas)} etiquetas de '{config.CLASE_OBJETIVO}'.")

        ancho_img = src_tif.width
        alto_img = src_tif.height
        coords_x = range(0, ancho_img - config.PATCH_SIZE, config.STRIDE)
        coords_y = range(0, alto_img - config.PATCH_SIZE, config.STRIDE)

        print(f"Generando parches de {config.PATCH_SIZE}x{config.PATCH_SIZE} con paso {config.STRIDE}...")
        pbar = tqdm(total=len(coords_x) * len(coords_y), desc="Procesando parches")
        
        contador_parches_guardados = 0
        contador_etiquetas_guardadas = 0

        for y in coords_y:
            for x in coords_x:
                pbar.update(1)
                ventana = Window(x, y, config.PATCH_SIZE, config.PATCH_SIZE)
                limites_ventana = rasterio.windows.bounds(ventana, src_tif.transform)
                patch_geom = box(*limites_ventana)
                etiquetas_en_patch = labels_filtradas[labels_filtradas.geometry.intersects(patch_geom)]

                guardar_este_parche = False
                if not etiquetas_en_patch.empty:
                    guardar_este_parche = True
                elif random.random() < config.PROB_GUARDAR_VACIO:
                    guardar_este_parche = True

                if guardar_este_parche:
                    contador_parches_guardados += 1
                    patch_imagen_array = src_tif.read(window=ventana)

                    if patch_imagen_array.shape[0] == 1:
                        patch_imagen_array = patch_imagen_array[0]
                        modo_pil = "L"
                    else:
                        patch_imagen_array = np.moveaxis(patch_imagen_array, 0, -1)
                        modo_pil = "RGB"

                    img = Image.fromarray(patch_imagen_array, mode=modo_pil)
                    nombre_parche = f"patch_{x}_{y}"
                    img.save(os.path.join(dir_imagenes, f"{nombre_parche}.png"))

                    ruta_txt = os.path.join(dir_etiquetas, f"{nombre_parche}.txt")
                    with open(ruta_txt, 'w') as f_txt:
                        if not etiquetas_en_patch.empty:
                            for _, etiqueta in etiquetas_en_patch.iterrows():
                                (g_xmin, g_ymin, g_xmax, g_ymax) = etiqueta.geometry.bounds
                                (px_fila_min, px_col_min) = src_tif.index(g_xmin, g_ymax)
                                (px_fila_max, px_col_max) = src_tif.index(g_xmax, g_ymin)

                                l_xmin = max(0, px_col_min - x)
                                l_ymin = max(0, px_fila_min - y)
                                l_xmax = min(config.PATCH_SIZE - 1, px_col_max - x)
                                l_ymax = min(config.PATCH_SIZE - 1, px_fila_max - y)

                                if l_xmax > l_xmin and l_ymax > l_ymin:
                                    yolo_bbox = normalizar_para_yolo((l_xmin, l_ymin, l_xmax, l_ymax), config.PATCH_SIZE, config.PATCH_SIZE)
                                    f_txt.write(f"0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")
                                    contador_etiquetas_guardadas += 1
        pbar.close()
        print("\n--- ¡Proceso Completado! ---")
        print(f"Se guardaron {contador_parches_guardados} parches de imagen.")
        print(f"Se encontraron y guardaron {contador_etiquetas_guardadas} etiquetas en total.")
        print(f"Tu dataset está listo en la carpeta: {config.RUTA_DATASET_PARCHES}")

if __name__ == "__main__":
    main()