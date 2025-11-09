from typing import List, Dict, Tuple, Optional
import os
import glob
import pydicom
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import pickle
import matplotlib.pyplot as plt
from datetime import datetime


class DicomManager:

    def __init__(self, folder_path: str):
        """Inicializa el gestor de archivos DICOM."""
        self.folder_path = folder_path
        self.dataset_list: List[pydicom.dataset.FileDataset] = []
        self.volume: Optional[np.ndarray] = None
        self.sorted_slices_info: List[Tuple[float, pydicom.dataset.FileDataset]] = []
        self.metadata_df: Optional[pd.DataFrame] = None

    def load_folder(self):
        files = sorted(glob.glob(os.path.join(self.folder_path, "*")))
        dicom_files = []
        
        for f in files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=False)
                dicom_files.append(ds)
            except Exception:
                continue
                
        if not dicom_files:
            raise FileNotFoundError(f"No se encontraron archivos DICOM en {self.folder_path}")
            
        self.dataset_list = dicom_files
        print(f"[DicomManager]  {len(self.dataset_list)} archivos DICOM cargados")


    def build_volume(self) :
       
        if not self.dataset_list:
            raise RuntimeError("No hay datasets. Ejecute load_folder() primero.")
            
        slices = []
        for ds in self.dataset_list:
            if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient:
                z = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, "SliceLocation") and ds.SliceLocation:
                z = float(ds.SliceLocation)
            else:
                z = 0.0
            slices.append((z, ds))
            
        slices.sort(key=lambda x: x[0])
        self.sorted_slices_info = slices
        
        try:
            pixel_arrays = [s[1].pixel_array for s in slices]
            self.volume = np.stack(pixel_arrays, axis=0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Error al reconstruir volumen: {e}")
            
        print(f"[DicomManager]  Volumen: {self.volume.shape} (z,y,x)")
        self._build_metadata_df()


    def _build_metadata_df(self):
    
        rows = []
        for z, ds in self.sorted_slices_info:
            rows.append({
                'Filename': getattr(ds, 'filename', ''),
                'InstanceNumber': getattr(ds, 'InstanceNumber', None),
                'StudyDate': getattr(ds, 'StudyDate', None),
                'StudyTime': getattr(ds, 'StudyTime', None),
                'SeriesTime': getattr(ds, 'SeriesTime', None),
                'Modality': getattr(ds, 'Modality', None),
                'StudyDescription': getattr(ds, 'StudyDescription', None),
                'PixelSpacing': str(getattr(ds, 'PixelSpacing', None)),
                'SliceThickness': getattr(ds, 'SliceThickness', None),
                'Rows': getattr(ds, 'Rows', None),
                'Columns': getattr(ds, 'Columns', None),
                'SliceLocation': z
            })
        self.metadata_df = pd.DataFrame(rows)

    def save_metadata_csv(self, out_path: str):
       
        if self.metadata_df is None:
            raise RuntimeError("Ejecute build_volume() primero.")
        self.metadata_df.to_csv(out_path, index=False)
        print(f"[DicomManager]  CSV guardado: '{out_path}'")

    def get_pixel_spacing(self) :

        if not self.sorted_slices_info:
            raise RuntimeError("Ejecute build_volume() primero.")
        ds = self.sorted_slices_info[0][1]
        ps = getattr(ds, "PixelSpacing", None)
        return (float(ps[0]), float(ps[1])) if ps else (1.0, 1.0)

    def get_slice_thickness(self):

        if not self.sorted_slices_info:
            raise RuntimeError("Ejecute build_volume() primero.")
        ds = self.sorted_slices_info[0][1]
        st = getattr(ds, "SliceThickness", None)
        return float(st) if st else 1.0
    
    def save_nifti(self, out_path: str):
        
        if self.volume is None:
            raise RuntimeError("Ejecute build_volume() primero.")
        px, py = self.get_pixel_spacing()
        pz = self.get_slice_thickness()
        affine = np.diag([px, py, pz, 1.0])
        vol_nib = np.transpose(self.volume, (2, 1, 0))
        nifti = nib.Nifti1Image(vol_nib, affine)
        nib.save(nifti, out_path)
        print(f"[DicomManager]  NIfTI guardado: '{out_path}'")
        
class EstudioImaginologico:
    
    def __init__(self, dicom_manager: DicomManager):

        if dicom_manager.volume is None:
            raise RuntimeError("DicomManager debe tener volumen construido.")
            
        self.dicom_manager = dicom_manager
        self.volume = dicom_manager.volume
        
        first_ds = dicom_manager.sorted_slices_info[0][1]
        self.study_date = getattr(first_ds, 'StudyDate', 'N/A')
        self.study_time = getattr(first_ds, 'StudyTime', 'N/A')
        self.study_modality = getattr(first_ds, 'Modality', 'N/A')
        self.study_description = getattr(first_ds, 'StudyDescription', 'N/A')
        self.series_time = getattr(first_ds, 'SeriesTime', 'N/A')
        self.duration_seconds = self._calc_duration_seconds(self.study_time, self.series_time)
        self.shape = self.volume.shape
        
        print(f"[EstudioImaginologico] {self.study_modality} - {self.study_description}")

    def _calc_duration_seconds(self, study_time: str, series_time: str) :
        
        try:
            if not study_time or not series_time:
                return None
            fmt = "%H%M%S"
            st = datetime.strptime(study_time.split('.')[0], fmt)
            se = datetime.strptime(series_time.split('.')[0], fmt)
            return (se - st).total_seconds()  
        except Exception:
            return None
 def show_orthogonal_slices(self, index_z: Optional[int] = None):
        
        from scipy.ndimage import zoom 

        z_len, y_len, x_len = self.volume.shape
        index_z = index_z if index_z else z_len // 2

        try:
            px, py = self.dicom_manager.get_pixel_spacing()
            pz = self.dicom_manager.get_slice_thickness()
        except RuntimeError:
            print("[EstudioImaginologico] Advertencia: No se pudieron obtener métricas espaciales. Usando (1,1,1).")
            px, py, pz = 1.0, 1.0, 1.0

        transversal = self.volume[index_z, :, :]
        sagital = self.volume[:, :, x_len // 2]
        coronal = self.volume[:, y_len // 2, :]

      
        scale_sagital_z = pz / py 
        sagital_interp = zoom(sagital, (scale_sagital_z, 1.0), order=3)
        
       
        scale_coronal_z = pz / px
        coronal_interp = zoom(coronal, (scale_coronal_z, 1.0), order=3)
        
       
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(transversal, cmap='gray', aspect='equal')
        axs[0].set_title('Transversal (Axial)', fontsize=12, fontweight='bold')

        axs[1].imshow(sagital_interp, cmap='gray', aspect='auto', origin='lower')
        axs[1].set_title('Sagital', fontsize=12, fontweight='bold')
        axs[1].axis('off')

        axs[2].imshow(coronal_interp, cmap='gray', aspect='auto', origin='lower')
        axs[2].set_title('Coronal', fontsize=12, fontweight='bold')
        axs[2].axis('off')

        plt.suptitle(f'{self.study_modality} - {self.study_description}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


    def _normalize_to_uint8(self, img: np.ndarray) :
        imgf = img.astype(np.float32)
        mn, mx = float(np.min(imgf)), float(np.max(imgf))
        if mx == mn:
            return np.zeros_like(imgf, dtype=np.uint8)
        norm = (imgf - mn) / (mx - mn) * 255.0
        return np.clip(norm, 0, 255).astype(np.uint8)

    def zoom_and_save(self):
        
        try:
            px, py = self.dicom_manager.get_pixel_spacing()
            print("\n" + "="*60)
            print("ZOOM - RECORTE Y REDIMENSIONAMIENTO")
            print("="*60)
            print("Secciones: 1=transversal, 2=sagital, 3=coronal")
            
            choice = input("Opción (1-3): ").strip()
            z_len, y_len, x_len = self.volume.shape
            
            if choice == '1':
                z_idx = int(input(f"Índice z [0-{z_len-1}, default={z_len//2}]: ") or z_len//2)
                img2d = self.volume[max(0, min(z_idx, z_len-1)), :, :]
                spacing_x, spacing_y = px, py
            elif choice == '2':
                x_idx = int(input(f"Índice x [0-{x_len-1}, default={x_len//2}]: ") or x_len//2)
                img2d = self.volume[:, :, max(0, min(x_idx, x_len-1))]
                spacing_x, spacing_y = self.dicom_manager.get_slice_thickness(), py
            elif choice == '3':
                y_idx = int(input(f"Índice y [0-{y_len-1}, default={y_len//2}]: ") or y_len//2)
                img2d = self.volume[:, max(0, min(y_idx, y_len-1)), :]
                spacing_x, spacing_y = self.dicom_manager.get_slice_thickness(), px
            else:
                print(" Opción inválida")
                return
            
            plt.figure(figsize=(8,8))
            plt.imshow(img2d, cmap='gray')
            plt.title('Imagen seleccionada (observe coordenadas)')
            plt.colorbar()
            plt.show()

            print(f"Dimensiones: {img2d.shape[1]}x{img2d.shape[0]} px")
            coords = input("Coordenadas recorte (x1,y1,x2,y2) [Enter=recorte centrado]: ").strip()
            
            if coords:
                x1, y1, x2, y2 = [int(c) for c in coords.split(',')]
            else:
                # recorte centrado por defecto
                h, w = img2d.shape
                x1, x2 = w//4, 3*w//4
                y1, y2 = h//4, 3*h//4
                print(f"Usando recorte automático: ({x1},{y1},{x2},{y2})")

            # Validaciones de coordenadas
            if x2 <= x1 or y2 <= y1:
                print(" Coordenadas inválidas: ancho y alto deben ser positivos.")
                return
            if x1 < 0 or y1 < 0 or x2 > img2d.shape[1] or y2 > img2d.shape[0]:
                print(" Coordenadas fuera del rango de la imagen.")
                return
            
            cropped = img2d[y1:y2, x1:x2]
            if cropped.size == 0:
                print(" El recorte está vacío.")
                return
            
            cropped_u8 = self._normalize_to_uint8(cropped)
            cropped_bgr = cv2.cvtColor(cropped_u8, cv2.COLOR_GRAY2BGR)
            
            width_mm = (x2 - x1) * spacing_x
            height_mm = (y2 - y1) * spacing_y
            dim_text = f"{width_mm:.1f}mm x {height_mm:.1f}mm"
            
            
            h, w = cropped_bgr.shape[:2]
            cv2.rectangle(cropped_bgr, (2, 2), (w-3, h-3), (0, 0, 255), 3)
            cv2.putText(cropped_bgr, dim_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            size_str = input("Redimensionar (ancho,alto) [Enter=omitir]: ").strip()
            if size_str:
                new_w, new_h = [int(x) for x in size_str.split(',')]
                resized = cv2.resize(cropped_bgr, (new_w, new_h))
            else:
                resized = cropped_bgr
            
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            axs[0].imshow(img2d, cmap='gray')
            axs[0].set_title('Original')
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                edgecolor='red', facecolor='none')
            axs[0].add_patch(rect)
            axs[0].axis('off')
            
            axs[1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            axs[1].set_title(f'Recorte - {dim_text}')
            axs[1].axis('off')
            plt.tight_layout()
            plt.show()
            fname = input("Nombre archivo [Enter=omitir]: ").strip()
            if fname:
                if not fname.endswith(('.png', '.jpg', '.jpeg')):
                    fname += '.png'
                cv2.imwrite(fname, resized)
                print(f" Guardado: '{fname}'")
                
        except Exception as e:
            print(f" Error: {e}")


    def segment_and_show(self):
        try:
            print("\n" + "="*60)
            print("SEGMENTACIÓN")
            print("="*60)
            
            choice = input("Sección (1=transversal, 2=sagital, 3=coronal): ").strip()
            z_len, y_len, x_len = self.volume.shape
            
            if choice == '1':
                idx = int(input(f"Índice z [0-{z_len-1}]: ") or z_len//2)
                img2d = self.volume[max(0, min(idx, z_len-1)), :, :]
            elif choice == '2':
                idx = int(input(f"Índice x [0-{x_len-1}]: ") or x_len//2)
                img2d = self.volume[:, :, max(0, min(idx, x_len-1))]
            elif choice == '3':
                idx = int(input(f"Índice y [0-{y_len-1}]: ") or y_len//2)
                img2d = self.volume[:, max(0, min(idx, y_len-1)), :]
            else:
                print(" Opción inválida")
                return
            
            img_u8 = self._normalize_to_uint8(img2d)
            
            methods = {
                '0': ('Binary', cv2.THRESH_BINARY),
                '1': ('Binary Inv', cv2.THRESH_BINARY_INV),
                '2': ('Truncate', cv2.THRESH_TRUNC),
                '3': ('ToZero', cv2.THRESH_TOZERO),
                '4': ('ToZero Inv', cv2.THRESH_TOZERO_INV),
                '5': ('Otsu', 'otsu')
            }
            
            print("\nMétodos de binarización:")
            for k, v in methods.items():
                print(f"  {k}) {v[0]}")
            
            method = input("Método: ").strip()
            if method not in methods:
                print(" Método inválido")
                return
            
            if methods[method][1] == 'otsu':
                _, thresh = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                thr = int(input("Umbral (0-255) [128]: ") or "128")
                _, thresh = cv2.threshold(img_u8, thr, 255, methods[method][1])
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].imshow(img_u8, cmap='gray')
            axs[0].set_title('Original')
            axs[0].axis('off')
            
            axs[1].imshow(thresh, cmap='gray')
            axs[1].set_title(f'Segmentado - {methods[method][0]}')
            axs[1].axis('off')
            plt.tight_layout()
            plt.show()
            
            if input("¿Guardar? (s/n): ").lower() == 's':
                fname = input("Nombre: ").strip()
                if fname:
                    if not fname.endswith('.png'):
                        fname += '.png'
                    cv2.imwrite(fname, thresh)
                    print(f"✓ Guardado: '{fname}'")
                    
        except Exception as e:
            print(f" Error: {e}")


    def morph_transform(self):
        try:
            print("\n" + "="*60)
            print("TRANSFORMACIÓN MORFOLÓGICA")
            print("="*60)
            print("Sección: 1=Transversal, 2=Sagital, 3=Coronal")

            choice = input("Opción (1-3): ").strip()
            z_len, y_len, x_len = self.volume.shape

           
            if choice == '1':
                idx = int(input(f"Índice z [0-{z_len-1}, default={z_len//2}]: ") or z_len//2)
                img2d = self.volume[max(0, min(idx, z_len-1)), :, :]
            elif choice == '2':
                idx = int(input(f"Índice x [0-{x_len-1}, default={x_len//2}]: ") or x_len//2)
                img2d = self.volume[:, :, max(0, min(idx, x_len-1))]
            elif choice == '3':
                idx = int(input(f"Índice y [0-{y_len-1}, default={y_len//2}]: ") or y_len//2)
                img2d = self.volume[:, max(0, min(idx, y_len-1)), :]
            else:
                print(" Opción inválida.")
                return
            img_u8 = self._normalize_to_uint8(img2d)
            if img_u8.size == 0:
                print(" Imagen vacía, no se puede procesar.")
                return

           
            k = int(input("Tamaño kernel (impar, ej: 3,5,7) [3]: ") or "3")
            if k < 1:
                print(" Tamaño de kernel inválido.")
                return
            if k % 2 == 0:
                k += 1  # forzar impar
                print(f"Ajustado a kernel {k}x{k} (impar)")

            
            ops = {
                0: ('Erode', cv2.MORPH_ERODE),
                1: ('Dilate', cv2.MORPH_DILATE),
                2: ('Open', cv2.MORPH_OPEN),
                3: ('Close', cv2.MORPH_CLOSE),
                4: ('Gradient', cv2.MORPH_GRADIENT),
                5: ('Tophat', cv2.MORPH_TOPHAT),
                6: ('Blackhat', cv2.MORPH_BLACKHAT)
            }

            print("\nOperaciones disponibles:")
            for k_op, (name, _) in ops.items():
                print(f"  {k_op}) {name}")

            op = int(input("Operación (0-6) [2]: ") or "2")
            if op not in ops:
                print(" Operación inválida.")
                return

       
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            res = cv2.morphologyEx(img_u8, ops[op][1], kernel)

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].imshow(img_u8, cmap='gray')
            axs[0].set_title('Original')
            axs[0].axis('off')

            axs[1].imshow(res, cmap='gray')
            axs[1].set_title(f'{ops[op][0]} - kernel={k}x{k}')
            axs[1].axis('off')
            plt.tight_layout()
            plt.show()

            if input("¿Guardar resultado? (s/n): ").strip().lower() == 's':
                fname = input("Nombre del archivo: ").strip()
                if fname:
                    if not fname.endswith('.png'):
                        fname += '.png'
                    cv2.imwrite(fname, res)
                    print(f" Imagen guardada: '{fname}'")

        except Exception as e:
            print(f" Error en transformación morfológica: {e}")
            
    def save_object(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[EstudioImaginologico]  Serializado: '{path}'")


            