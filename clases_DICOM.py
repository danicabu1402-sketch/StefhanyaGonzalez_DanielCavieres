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
