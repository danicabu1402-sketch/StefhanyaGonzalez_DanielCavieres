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

    def load_folder(self) -> None:
        """Carga todos los archivos DICOM válidos desde la carpeta especificada."""
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
        print(f"[DicomManager] ✓ {len(self.dataset_list)} archivos DICOM cargados")
