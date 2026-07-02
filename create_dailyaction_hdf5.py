import os
import glob
import random
import numpy as np
import h5py
from tqdm import tqdm

def read_custom_aedat_v2(filepath):
    """Lector nativo ultrarrápido bi-endian para DVS128."""
    with open(filepath, 'rb') as file:
        while True:
            pos = file.tell()
            line = file.readline()
            if not line.startswith(b'#'):
                file.seek(pos)
                break
        data = file.read()
        
    raw = np.frombuffer(data, dtype=np.dtype('>u4'))
    if len(raw) < 2:
        return None
        
    address = raw[0::2]
    t = raw[1::2]
    
    # Decodificación DVS128
    x = np.bitwise_and(address >> 1, 0x7F).astype(np.int16)
    y = np.bitwise_and(address >> 8, 0x7F).astype(np.int16)
    p = np.bitwise_and(address, 1).astype(np.int16)
    
    # [Y-INVERSION] Se corrige la orientación de cabeza observada visualmente.
    # El sensor DVS128 tiene Y=127 como maximo.
    y = 127 - y
    
    events = np.zeros(len(x), dtype=[('x', np.int16), ('y', np.int16), ('p', np.int16), ('t', np.int64)])
    events['x'] = x
    events['y'] = y
    events['p'] = p
    events['t'] = t.astype(np.int64)
    
    return events


def create_dailyaction_hdf5(dataset_dir, out_h5="dailyaction_ds1.h5", test_ratio=0.2):
    print(f"[DailyAction-DVS] Iniciando ingesta masiva desde: {dataset_dir}")
    
    # Identificar todas las subcarpetas (clases)
    class_folders = sorted([d.name for d in os.scandir(dataset_dir) if d.is_dir()])
    
    class_map = {name: idx for idx, name in enumerate(class_folders)}
    class_names = [name for name in class_folders]
    
    print(f"Detectadas {len(class_folders)} Clases: {class_map}")
    
    # Acumular archivos
    train_files = []
    test_files = []
    
    # [LOSO SPLIT]
    # We have 15 subjects total (cc,gh,hk,hr,jc,jf,kx,ls,ps,sh,sy,xc,xd,yn,zh)
    # Reserving 3 subjects exactly matches the 20% test ratio.
    LOSO_TEST_SUBJECTS = {'ps', 'sy', 'xd'}
    print(f"[LOSO] Sujetos rígidamente reservados para Testeo: {LOSO_TEST_SUBJECTS}")
    
    total_files = 0
    for cls_name in class_folders:
        cls_dir = os.path.join(dataset_dir, cls_name)
        file_list = sorted(glob.glob(os.path.join(cls_dir, "*.aedat"))) # DailyAction usa .aedat v2
        
        n_train_cls = 0
        n_test_cls = 0
        
        for f in file_list:
            basename = os.path.basename(f)
            subject = basename[:2].lower()
            
            if subject in LOSO_TEST_SUBJECTS:
                test_files.append((f, class_map[cls_name]))
                n_test_cls += 1
            else:
                train_files.append((f, class_map[cls_name]))
                n_train_cls += 1
                
        total_files += len(file_list)
        print(f"  Clase [{cls_name.upper()}] -> Train: {n_train_cls} | Test: {n_test_cls}")
        
    print(f"\n[INFO] Total a procesar: {total_files} grabaciones. Splitting LOSO terminado.")
    
    W, H, P = 128, 128, 2
    
    # Escribir el H5
    with h5py.File(out_h5, 'w') as f:
        # Metadatos globales
        f.attrs['classes'] = class_names
        f.attrs['sensor_size'] = (W, H, P)
        f.attrs['num_classes'] = len(class_names)
        
        grp_train = f.create_group('train')
        grp_test = f.create_group('test')
        
        print("\nEmpacando subset TRAIN...")
        for i, (filepath, label) in enumerate(tqdm(train_files)):
            events = read_custom_aedat_v2(filepath)
            if events is not None and len(events) > 0:
                ds = grp_train.create_dataset(str(i), data=events, compression='gzip')
                ds.attrs['label'] = label
                
        print("\nEmpacando subset TEST...")
        for i, (filepath, label) in enumerate(tqdm(test_files)):
            events = read_custom_aedat_v2(filepath)
            if events is not None and len(events) > 0:
                ds = grp_test.create_dataset(str(i), data=events, compression='gzip')
                ds.attrs['label'] = label

    print(f"\n[DONE] Construcción del Tensor HDF5 completada -> {out_h5}")


if __name__ == "__main__":
    SRC_DIR = r"C:\Users\admin\Downloads\DailyAction-DVS\DailyAction-DVS"
    OUT_FILE = "dailyaction_ds_loso.h5"
    create_dailyaction_hdf5(SRC_DIR, OUT_FILE)
