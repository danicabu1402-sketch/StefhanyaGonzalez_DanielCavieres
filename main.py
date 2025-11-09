import os
from clases_DICOM import DicomManager, EstudioImaginologico

def listar_datos():
    base = "datos"
    if not os.path.exists(base):
        print("Carpeta 'datos' no encontrada. Cree una carpeta 'datos' con subcarpetas PPMI, Sarcoma, T2.")
        return
    for d in os.listdir(base):
        print("-", d)

def menu():
    estudios = {}  
    dicom_managers = {}
    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1) Cargar carpeta DICOM (crear DicomManager)")
        print("2) Reconstruir volumen 3D desde una carpeta cargada")
        print("3) Crear EstudioImaginologico desde una carpeta reconstruida")
        print("4) Mostrar 3 cortes (coronal, sagital, transversal) de un estudio")
        print("5) Mostrar/guardar recorte (zoom) de estudio")
        print("6) Segmentación (binarización) de un corte")
        print("7) Transformación morfológica de un corte")
        print("8) Convertir carpeta DICOM a NIfTI")
        print("9) Guardar metadata CSV del DicomManager")
        print("10) Listar objetos creados")
        print("11) Guardar objeto EstudioImaginologico (serializar)")
        print("0) Salir")
        choice = input("Opción: ").strip()
        if choice == '1':
            path = input("Ruta de la carpeta DICOM: ").strip()
            key = os.path.basename(path.rstrip('/\\'))
            dm = DicomManager(path)
            try:
                dm.load_folder()
                dicom_managers[key] = dm
            except Exception as e:
                print("Error:", e)
        elif choice == '2':
            key = input("Clave del DicomManager a reconstruir (nombre de carpeta): ").strip()
            if key not in dicom_managers:
                print("Clave no encontrada.")
                continue
            try:
                dicom_managers[key].build_volume()
            except Exception as e:
                print("Error:", e)
        elif choice == '3':
            key = input("Clave del DicomManager para crear Estudio: ").strip()
            if key not in dicom_managers:
                print("No existe ese DicomManager.")
                continue
            dm = dicom_managers[key]
            if dm.volume is None:
                print("Construya primero el volumen (opción 2).")
                continue
            estudio_key = input("Nombre para registrar el EstudioImaginologico: ").strip()
            estudio = EstudioImaginologico(dm)
            estudios[estudio_key] = estudio
            print(f"Estudio registrado con clave '{estudio_key}'")
        elif choice == '4':
            k = input("Clave del estudio: ").strip()
            if k not in estudios:
                print("Estudio no encontrado.")
                continue
            estudios[k].show_orthogonal_slices()
        elif choice == '5':
            k = input("Clave del estudio: ").strip()
            if k not in estudios:
                print("Estudio no encontrado.")
                continue
            estudios[k].zoom_and_save()
        elif choice == '6':
            k = input("Clave del estudio: ").strip()
            if k not in estudios:
                print("Estudio no encontrado.")
                continue
            estudios[k].segment_and_show()
        elif choice == '7':
            k = input("Clave del estudio: ").strip()
            if k not in estudios:
                print("Estudio no encontrado.")
                continue
            estudios[k].morph_transform()
        elif choice == '8':
            key = input("Clave del DicomManager a convertir: ").strip()
            if key not in dicom_managers:
                print("DicomManager no encontrado.")
                continue
            outname = input("Nombre archivo salida NIfTI (ej: salida.nii.gz): ").strip()
            dicom_managers[key].save_nifti(outname)
        elif choice == '9':
            key = input("Clave del DicomManager: ").strip()
            if key not in dicom_managers:
                print("DicomManager no encontrado.")
                continue
            outcsv = input("Nombre CSV salida (ej: metadata.csv): ").strip()
            dicom_managers[key].save_metadata_csv(outcsv)
        elif choice == '10':
            print("DicomManagers cargados:", list(dicom_managers.keys()))
            print("Estudios creados:", list(estudios.keys()))
        elif choice == '11':
            k = input("Clave del estudio a serializar: ").strip()
            if k not in estudios:
                print("Estudio no encontrado.")
                continue
            fname = input("Nombre archivo .pkl (ej: estudio1.pkl): ").strip()
            estudios[k].save_object(fname)
        elif choice == '0':
            print("Saliendo.")
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    print("Script de implementación para proyecto DICOM.")
    print("Asegúrese de tener las carpetas con datos: datos/PPMI, datos/Sarcoma, datos/T2")
    menu()
