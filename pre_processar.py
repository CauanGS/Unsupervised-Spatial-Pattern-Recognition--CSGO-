import os, json, lzma, glob
import numpy as np
import json 

def make_formation_image(t_positions, image_size=128):
    img = np.zeros((image_size, image_size), dtype=np.float32)
    T_COLOR = 1.0  

    for (x, y) in t_positions:
        cx = int(x * (image_size - 1))
        cy = int(y * (image_size - 1))
        y_min, y_max = np.clip(cy - 1, 0, image_size), np.clip(cy + 1, 0, image_size)
        x_min, x_max = np.clip(cx - 1, 0, image_size), np.clip(cx + 1, 0, image_size)
        img[y_min:y_max, x_min:x_max] = T_COLOR
        
    return img[..., None] 

PATH_DATASET = r"C:\Users\cauan\Documents\Projeto Machine Learning\esta-main\esta-main\data\online"
MAPA_ALVO = "de_mirage"
MAX_MATCHES = 900
MAX_ROUNDS_PER_MATCH = 20
IMAGE_SIZE = 64
OUTPUT_FILE_X = "mirage_data_X.npy"
OUTPUT_FILE_Y = "mirage_data_Y.npy"
OUTPUT_FILE_COORDS = "mirage_data_Coords.json"

print(f"Iniciando pré-processamento para '{MAPA_ALVO}' (Foco em T)...")
print(f"Buscando arquivos em: {PATH_DATASET}")

X_list, X_coords_raw, Y_winner = [], [], []

search_pattern = f"{PATH_DATASET}/**/*.xz"
demo_files = glob.glob(search_pattern, recursive=True)
demo_files = demo_files[:MAX_MATCHES]

print(f"Total de arquivos .xz encontrados (antes do filtro): {len(demo_files)}")

frames_added_total = 0
files_skipped_map = 0

for i, fpath in enumerate(demo_files):
    if (i+1) % 50 == 0:
        print(f"Processando arquivo {i+1}/{len(demo_files)}...")
        
    try:
        with lzma.open(fpath, mode='rt') as f:
            demo = json.load(f)
    except Exception as e:
        continue 
        
    if demo.get("mapName") != MAPA_ALVO:
        files_skipped_map += 1
        continue

    for rnd in demo.get("gameRounds", []):
        round_winner = rnd.get("winningSide") 
        if round_winner not in ["T", "CT"]:
            continue 
            
        frames = rnd.get("frames", [])
        
        target_frame_index = 30
        
        if len(frames) <= target_frame_index:
            continue 

        frame = frames[target_frame_index] 
        
        players_t = frame.get("t", {}).get("players", [])
        players_ct = frame.get("ct", {}).get("players", [])
        
        if players_t is None: players_t = []
        if players_ct is None: players_ct = []
        
        if len(players_t) < 5 or len(players_ct) < 5:
            continue 

        coords_t = [(p["x"], p["y"]) for p in players_t[:5]]
        coords_ct = [(p["x"], p["y"]) for p in players_ct[:5]]

        xmin, xmax = -4000.0, 4000.0
        ymin, ymax = -4000.0, 4000.0
        coords_norm_t = [((x - xmin)/(xmax - xmin), (y - ymin)/(ymax - ymin)) for x,y in coords_t]
        
        img = make_formation_image(coords_norm_t, image_size=IMAGE_SIZE)
        
        X_list.append(img) 
        X_coords_raw.append({"t": coords_t, "ct": coords_ct})
        Y_winner.append(round_winner) 
        frames_added_total += 1

print(f"\n--- Processamento Concluído ---")
print(f"Arquivos pulados (mapa errado): {files_skipped_map}")
print(f"Total de frames (imagens) coletados: {frames_added_total}")

if frames_added_total > 0:
    X = np.stack(X_list, axis=0) 
    Y = np.array(Y_winner)
    
    np.save(OUTPUT_FILE_X, X)
    np.save(OUTPUT_FILE_Y, Y)
    with open(OUTPUT_FILE_COORDS, 'w') as f:
        json.dump(X_coords_raw, f)

    print(f"\n--- SUCESSO! ---")
    print(f"Dados de imagem (APENAS T) salvos em: {OUTPUT_FILE_X} (Shape: {X.shape})")
    print(f"Dados de vencedor salvos em: {OUTPUT_FILE_Y}")
    print(f"Dados de coordenadas (T e CT) salvos em: {OUTPUT_FILE_COORDS}")
else:
    print("Nenhum dado foi encontrado. Nenhum arquivo foi salvo.")