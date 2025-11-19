import os, json
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)

INPUT_FILE_X = "mirage_data_X.npy"
INPUT_FILE_Y = "mirage_data_Y.npy"
INPUT_FILE_COORDS = "mirage_data_Coords.json"
MAPA_ALVO = "de_mirage" 

try:
    X = np.load(INPUT_FILE_X)
    Y_winner = np.load(INPUT_FILE_Y)
    with open(INPUT_FILE_COORDS, 'r') as f:
        X_coords_raw = json.load(f)

    print(f"Dados pré-processados carregados com sucesso.")
    
    unique_winners = np.unique(Y_winner)
    print(f"Valores únicos encontrados em Y_winner: {unique_winners}")
    
    print(f"Dataset real carregado: {X.shape}") 
    print(f"Rounds (amostras) carregados: {len(Y_winner)}")
    
    image_size = X.shape[1]

except FileNotFoundError:
    print(f"ERRO: Arquivos de dados (.npy, .json) não encontrados.")
    exit()
except Exception as e:
    print(f"Erro ao carregar os arquivos de dados: {e}")
    exit()

ENCODER_FILE = "encoder_model.keras" 
latent_dim = 64
INPUT_SHAPE = (image_size, image_size, 1) 

if os.path.exists(ENCODER_FILE):
    print(f"Carregando modelo 'encoder' salvo de {ENCODER_FILE}...")
    encoder = keras.models.load_model(ENCODER_FILE)

else:
    print(f"Arquivo {ENCODER_FILE} não encontrado.")
    
    def build_cae(input_shape=INPUT_SHAPE, latent_dim=64):
        encoder_inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
        x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
        shape_before_flatten = keras.backend.int_shape(x)[1:]
        x = layers.Flatten()(x)
        latent = layers.Dense(latent_dim, name='latent_vector')(x)
        encoder = keras.Model(encoder_inputs, latent, name='encoder')

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(np.prod(shape_before_flatten), activation='relu')(latent_inputs)
        x = layers.Reshape(shape_before_flatten)(x)
        x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
        decoder = keras.Model(latent_inputs, x, name='decoder')

        ae_inputs = encoder_inputs
        encoded = encoder(ae_inputs)
        reconstructed = decoder(encoded)
        autoencoder = keras.Model(ae_inputs, reconstructed, name='autoencoder')
        return encoder, decoder, autoencoder

    encoder, decoder, autoencoder = build_cae(input_shape=INPUT_SHAPE, latent_dim=latent_dim)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    history = autoencoder.fit(X, X, epochs=3, batch_size=32, validation_split=0.1, verbose=2)
  
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.legend(); plt.title(f"Curva de perda (CAE) - {MAPA_ALVO}"); plt.show()
    
    print(f"Salvando o modelo 'encoder' treinado em {ENCODER_FILE}...")
    encoder.save(ENCODER_FILE)
    print("Modelo salvo com sucesso.")

print("\nCalculando o 'eps' ideal para o DBSCAN...")

embeddings = encoder.predict(X, batch_size=64)
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(embeddings)

pca_components = min(30, X.shape[0] - 1) 
if pca_components <= 1: pca_components = 2
pca = PCA(n_components=pca_components, random_state=42)
emb_pca = pca.fit_transform(emb_scaled)

perplexity_value = min(30, X.shape[0] - 1)
if perplexity_value <= 1: perplexity_value = 2
tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=perplexity_value)
emb_tsne = tsne.fit_transform(emb_pca) 

print("Calculando o 'eps' ideal para os dados 2D...")

min_samples = 5 
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(emb_tsne) 
distances, indices = neighbors_fit.kneighbors(emb_tsne)
distances = np.sort(distances[:, min_samples-1], axis=0)

plt.figure(figsize=(10,5))
plt.plot(distances)
plt.title(f"Gráfico K-distance (para dados 2D do t-SNE)")
plt.xlabel("Índice de Pontos (ordenado por distância)")
plt.ylabel(f"Distância para o {min_samples}º vizinho (eps)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

VALOR_EPS_DO_GRAFICO = 2  
print(f"Executando DBSCAN nos dados 2D com eps={VALOR_EPS_DO_GRAFICO}...")

dbscan = DBSCAN(eps=VALOR_EPS_DO_GRAFICO, min_samples=5)
cluster_labels = dbscan.fit_predict(emb_tsne) 

plt.figure(figsize=(7,5))
unique_clusters_for_plot = np.unique(cluster_labels)
n_clusters_found = len([c for c in unique_clusters_for_plot if c != -1])
print(f"DBSCAN encontrou {n_clusters_found} clusters táticos puros.")

for c in unique_clusters_for_plot:
    idxs = (cluster_labels == c)
    if c == -1:
        plt.scatter(emb_tsne[idxs,0], emb_tsne[idxs,1], s=5, color='gray', label=f"Ruído (n={np.sum(idxs)})")
    else:
        plt.scatter(emb_tsne[idxs,0], emb_tsne[idxs,1], s=10, label=f"Cluster {c} (n={np.sum(idxs)})")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(f"Clusters de formações T (DBSCAN) - {MAPA_ALVO}")
plt.tight_layout()
plt.show()

print("Analisando resultados táticos")

results_df = pd.DataFrame({
    'cluster': cluster_labels, 
    'winner': Y_winner
})

if 'T' in unique_winners:
    results_df['T_Win'] = (results_df['winner'] == 'T').astype(int)
    print("Analisando para vencedor = 'T' (Maiúsculo)")
elif 't' in unique_winners:
    results_df['T_Win'] = (results_df['winner'] == 't').astype(int)
    print("Analisando para vencedor = 't' (Minúsculo)")
else:
    results_df['T_Win'] = 0 
    print(f"Atenção: Vencedor 'T' ou 't' não encontrado em {unique_winners}. Taxa de vitória será 0.")

cluster_analysis = results_df.groupby('cluster')['T_Win'].agg(
    T_Win_Rate='mean',
    Count='count'
).reset_index()

cluster_analysis['T_Win_Rate'] = cluster_analysis['T_Win_Rate'] * 100

print("\n--- Análise Tática por Cluster (Lado T) ---")
print(cluster_analysis.to_string(index=False, float_format="%.1f%%"))

cluster_analysis_sem_ruido = cluster_analysis[cluster_analysis['cluster'] != -1]
top_5_clusters = cluster_analysis_sem_ruido.sort_values(by='T_Win_Rate', ascending=False).head(5)
top_5_cluster_ids = top_5_clusters['cluster'].tolist() 

print(f"\n--- Top 5 Clusters Táticos (por Win Rate) ---")
print(top_5_clusters.to_string(index=False, float_format="%.1f%%"))

if top_5_clusters.empty:
    print("Nenhum cluster tático puro foi encontrado.")
else:
    plt.figure(figsize=(10, 5))
    bar_labels = top_5_clusters['cluster'].astype(str)
    bars = plt.bar(bar_labels, top_5_clusters['T_Win_Rate'], 
                   color='lightcoral', edgecolor='black')

    plt.title(f'Top 5 Clusters Táticos (por Win Rate) em {MAPA_ALVO}', fontsize=14)
    plt.xlabel('Cluster Identificado (Top 5)', fontsize=12)
    plt.ylabel('Taxa de Vitória T (%)', fontsize=12)
    plt.ylim(0, 100)
    
    for i, bar in enumerate(bars):
        row = top_5_clusters.iloc[i]
        rate = row['T_Win_Rate']
        count = row['Count']
        
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1, 
                 f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')
        plt.text(bar.get_x() + bar.get_width()/2.0, 5, 
                 f'(n={count} rounds)', ha='center', va='bottom', fontsize=9, color='black')

    plt.axhline(y=50, color='gray', linestyle='--', linewidth=1)
    if len(bars) > 0:
        plt.text(len(bars)-0.5, 51, '50% (Equilíbrio)', color='gray')
    plt.show()

SCALE = 0.18
X_OFFSET = 700
Y_OFFSET = 420

def game_to_image_coords_manual(game_x, game_y):
    px = (game_x * SCALE) + X_OFFSET
    py = (game_y * -SCALE) + Y_OFFSET 
    return px, py

map_image_path = os.path.join("maps", f"{MAPA_ALVO}.jpg")
try:
    map_img = plt.imread(map_image_path)
except FileNotFoundError:
    print(f"ERRO: Imagem do mapa não encontrada em '{map_image_path}'")
    map_img = np.zeros((714, 839, 3)) 
except Exception as e:
    print(f"Erro ao ler a imagem {map_image_path}: {e}")
    map_img = np.zeros((714, 839, 3))

try:
    IMG_HEIGHT, IMG_WIDTH, _ = map_img.shape
except ValueError:
    IMG_HEIGHT, IMG_WIDTH = 714, 839 
print(f"Plotando em cima da imagem de {IMG_WIDTH}x{IMG_HEIGHT} pixels.")

n_show = 5

if not 'top_5_cluster_ids' in locals() or not top_5_cluster_ids:
    print("Nenhum cluster tático puro encontrado para visualizar.")
else:
    print(f"Visualizando os Top 5 clusters: {top_5_cluster_ids}")
    
    fig = plt.figure(figsize=(15, 3 * len(top_5_cluster_ids))) 
    
    for r, c in enumerate(top_5_cluster_ids):
        idxs = np.where(cluster_labels == c)[0]
        
        if len(idxs) == 0:
            continue 

        chosen = np.random.choice(idxs, size=min(n_show, len(idxs)), replace=False)
        
        for j, idx in enumerate(chosen):
            ax = fig.add_subplot(len(top_5_cluster_ids), n_show, r*n_show + j + 1)
            
            ax.imshow(map_img) 
            
            coords_data = X_coords_raw[idx] 
            
            for (x, y) in coords_data["t"]:
                px, py = game_to_image_coords_manual(x, y)
                ax.plot(px, py, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=0.5)
                
            for (x, y) in coords_data["ct"]:
                px, py = game_to_image_coords_manual(x, y)
                ax.plot(px, py, 'o', color='blue', markersize=4, markeredgecolor='white', markeredgewidth=0.5)

            ax.set_axis_off() 
            ax.set_ylim(IMG_HEIGHT, 0) 
            ax.set_xlim(0, IMG_WIDTH)  
            
            if j == 0:
                win_rate_series = top_5_clusters.loc[top_5_clusters['cluster'] == c, 'T_Win_Rate']
                win_rate_str = f"{win_rate_series.iloc[0]:.1f}%" if not win_rate_series.empty else ""
                ax.set_title(f'Rank {r+1} - Cluster {c} (Win Rate: {win_rate_str})', loc='left', fontsize=10)

    plt.suptitle(f"Top 5 Formações Táticas (por Win Rate) em {MAPA_ALVO}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

print("Script finalizado.")