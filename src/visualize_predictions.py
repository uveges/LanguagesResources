import os
from enum import Enum
from typing import Union

# from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline

LABELS_COLUMN = "Label"       # Gold standard
# LABELS_COLUMN = "Predicted"     # huBERTPlain prediction
SEED = 42
fig_folder = "/home/istvanu/PycharmProjects/comprehensibility_FULL/figures/"
fig_name = "huBERT_original.png"
# fig_name = "huBERTPlain_Gold_Standard.png"
# fig_name = "huBERTPlain_Predictions.png"

def main():
    ######################### VECTORIZE DATA ###########################################################################
    # test_data = pd.read_excel("/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/COMPREHENSIBILITY_test.xlsx")
    # model_string = "uvegesistvan/huBERTPlain"
    # folder_to_save_vectors = "/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/BERT_vectors"

    # model_string = "SZTAKI-HLT/hubert-base-cc"
    # folder_to_save_vectors = "/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/huBERT_vectors"
    # vectorize_data_with_BERT(df=test_data, model_string=model_string, vectors_folder=folder_to_save_vectors)
    ######################### VISUALIZE VECTORS ########################################################################
    """
    Ábrák:
    LABELS_COLUMN = "Label" + sima huBERT --> Hogy néztek ki a szövegeket repr. vek.-k fine-tune előtt
    LABELS_COLUMN = "Label" + huBERTPlain --> Milyen lett a vektortér fine-tune után -- helyes címkézés
    LABELS_COLUMN = "Predicted" + huBERTPlain --> Milyen a modell valós címkézése (mennyire egyezik az előzővel)
    """

    # vectors = "/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/huBERT_vectors"   # huBERT
    # vectors = "/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/BERT_vectors"     # huBERTPlain
    # df = pd.read_excel("/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/COMPREHENSIBILITY_test.xlsx")

    # vizualizalas_pca(df=df, folder_path=vectors)
    # vizualizalas_pca_custom_colors(df=df, folder_path=vectors)

    ######################## PREDICTIONS WITH BERT #####################################################################
    # model_string = "uvegesistvan/huBERTPlain"
    # df = pd.read_excel("/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/COMPREHENSIBILITY_test.xlsx")
    # print(predict_with_bert(df=df, model_name=model_string))

    ######################## CLASSIFICATION REPORT #####################################################################
    # df = pd.read_excel("/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/COMPREHENSIBILITY_test.xlsx")
    # create_classification_report(df=df)

    ######################## COLOR BASED ON "SCORE" ####################################################################

    """ Score: The corresponding probability! --> https://huggingface.co/transformers/v4.10.1/main_classes/pipelines.html
    """

    # vectors = "/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/BERT_vectors"     # huBERTPlain
    # df = pd.read_excel("/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/COMPREHENSIBILITY_test.xlsx")
    # vizualizalas_pca_with_scores(df=df, folder_path=vectors)

    # Hogyan oszlanak meg a predikciók Score-jai?
    df = pd.read_excel("/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/COMPREHENSIBILITY_test.xlsx")
    # Gold Standard címkék
    fig_name = "GS_"
    plot_histogram_by_label_5_percent(df=df, label_column="Label", figure_name=fig_name)
    # Prediktált címkék
    fig_name = "Predicted_"
    plot_histogram_by_label_5_percent(df=df, label_column="Predicted", figure_name=fig_name)


def predict_with_bert(df, model_name):
    predictions = []

    # A modellhez tartozó tokenizer betöltése
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Hugging Face pipeline inicializálása a modellhez
    nlp = pipeline("text-classification", model=model_name, truncation=True, max_length=512)

    for index, row in tqdm(df.iterrows()):
        prediction = nlp(row['Text'])
        predictions.append(prediction)

    return predictions


def vizualizalas_pca_with_scores(df, folder_path):
    vectors = []  # Vektorok tárolása
    labels = []  # Címkék tárolása
    scores = []  # Skórok tárolása

    # DataFrame soronkénti bejárása
    for index, row in df.iterrows():
        file_path = os.path.join(folder_path, f"{row['id']}.npy")
        if os.path.exists(file_path):
            vector = np.load(file_path)
            vectors.append(vector.flatten())  # Vektor hozzáadása
            labels.append(row['Predicted'])  # Címke hozzáadása
            scores.append(row['Score'])  # Skór hozzáadása

    # Standardizálás
    vectors_standardized = StandardScaler().fit_transform(vectors)

    # PCA alkalmazása 2 dimenzióra csökkentéshez
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(vectors_standardized)

    # Vizualizáció előkészítése
    finalDf = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    finalDf['Label'] = labels
    finalDf['Score'] = scores

    # Színskála létrehozása a skórok alapján
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(scores), vmax=max(scores))

    # Vizualizáció
    fig, ax = plt.subplots()
    scatter = ax.scatter(finalDf['PC1'], finalDf['PC2'], c=finalDf['Score'], cmap=cmap, norm=norm)

    # Színskála hozzáadása
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Score')

    ax.grid()
    plt.savefig(os.path.join(fig_folder, "huBERTPlain prediction scores"))
    plt.show()

def vizualizalas_pca_custom_colors(df, folder_path):
    vectors = []  # Itt tároljuk a vektorokat
    labels = []  # És a hozzájuk tartozó címkéket

    # DataFrame soronkénti bejárása
    for index, row in df.iterrows():
        file_path = os.path.join(folder_path, f"{row['id']}.npy")
        if os.path.exists(file_path):
            vector = np.load(file_path)
            vectors.append(vector.flatten())  # Vektor hozzáadása, lapítva
            labels.append(row[LABELS_COLUMN])  # Címke hozzáadása

    # Standardizálás
    vectors_standardized = StandardScaler().fit_transform(vectors)

    # PCA alkalmazása 2 dimenzióra csökkentéshez
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(vectors_standardized)

    finalDf = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    finalDf['Label'] = labels

    color_map = {
        'original': 'red',
        'rephrased': 'green',
    }

    fig, ax = plt.subplots()
    for label, color in color_map.items():
        indicesToKeep = finalDf[LABELS_COLUMN] == label
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1'], finalDf.loc[indicesToKeep, 'PC2'], c=color, s=50, label=label)
    ax.legend()
    ax.grid()
    plt.savefig(os.path.join(fig_folder, fig_name))
    plt.show()


def vizualizalas_pca(df: pd.DataFrame, folder_path: str):
    vectors = []  # Itt tároljuk a vektorokat
    labels = []  # És a hozzájuk tartozó címkéket

    # DataFrame soronkénti bejárása
    for index, row in df.iterrows():
        file_path = os.path.join(folder_path, f"{row['id']}.npy")
        if os.path.exists(file_path):
            vector = np.load(file_path)
            vectors.append(vector.flatten())  # Vektor hozzáadása, lapítva
            labels.append(row[LABELS_COLUMN])  # Címke hozzáadása

    # Standardizálás
    vectors_standardized = StandardScaler().fit_transform(vectors)

    # PCA alkalmazása 2 dimenzióra csökkentéshez
    pca = PCA(n_components=2, random_state=SEED)
    principal_components = pca.fit_transform(vectors_standardized)

    # Vizualizáció előkészítése
    finalDf = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    finalDf[LABELS_COLUMN] = labels

    # Egyedi címkék lekérdezése
    unique_labels = list(set(labels))

    # Vizualizáció
    fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    color_map = {
        'original': 'red',
        'rephrased': 'green',
    }

    for label, color in zip(unique_labels, color_map.values()):
        indicesToKeep = finalDf[LABELS_COLUMN] == label
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                   , finalDf.loc[indicesToKeep, 'PC2']
                   , c=[color]
                   , s=50
                   , label=label)
    ax.legend(unique_labels)
    ax.grid()
    plt.savefig(os.path.join(fig_folder, fig_name))
    plt.show()


def create_classification_report(df):
    # Ellenőrizzük, hogy a szükséges oszlopok megtalálhatóak-e a DataFrame-ben
    if 'Label' not in df.columns or 'Predicted' not in df.columns:
        raise ValueError("A DataFrame-nek tartalmaznia kell egy 'Label' és egy 'Predicted' oszlopot.")

    # Készítünk egy klasszifikációs jelentést a valódi és prediktált címkékre
    report = classification_report(df['Label'], df['Predicted'])

    print("Classification Report:\n", report)


def vectorize_data_with_BERT(df: pd.DataFrame, model_string: str, vectors_folder: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    model = AutoModel.from_pretrained(model_string)

    for index, row in tqdm(df.iterrows()):
        inputs = tokenizer(row['Text'].strip(), return_tensors='pt', padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        file_name = f"{row['id']}.npy"
        file_path = os.path.join(vectors_folder, file_name)
        np.save(file_path, cls_embedding)


def plot_histogram_by_label_5_percent(df, label_column, figure_name):
    # Ellenőrizzük, hogy a szükséges oszlopok léteznek-e a DataFrame-ben
    if 'Score' not in df.columns or label_column not in df.columns:
        raise ValueError("A DataFrame-nek tartalmaznia kell egy 'Score' és egy 'Label' oszlopot.")

    # Szűrjük a DataFrame-et, hogy csak a 0.5 és 1 közötti értékek maradjanak
    filtered_df = df[(df['Score'] >= 0.5) & (df['Score'] <= 1)]

    # Egyedi címkék lekérdezése
    labels = filtered_df[label_column].unique()

    # Minden címke szerint külön hisztogram készítése
    for label in labels:
        plt.figure()  # Új ábra létrehozása minden címkéhez
        # Adott címkéjű sorok kiválasztása
        subset = filtered_df[filtered_df[label_column] == label]

        # Hisztogram létrehozása a megadott intervallumokra (0.5 és 1 között 5%-os lépésekkel)
        counts, bins, patches = plt.hist(subset['Score'],
                                         bins=np.linspace(0.5, 1, 11),
                                         edgecolor='black',
                                         color='red' if label == 'original' else 'green')

        # Az összes érték megszámolása a csoporton belül
        total = len(subset['Score'])

        # Az egyes oszlopok felett százalékos értékek megjelenítése
        for count, patch in zip(counts, patches):
            # Százalék kiszámítása
            percentage = (count / total) * 100 if total > 0 else 0
            # Százalék megjelenítése az oszlop felett
            plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f'{percentage:.1f}%',
                     ha='center', va='bottom')

        # Cím és tengelyek címkézése
        plt.title(f'Histogram of Score Values for Label: {label}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        fig_name = figure_name + label
        plt.savefig(os.path.join(fig_folder, fig_name))
        plt.show()


if __name__ == '__main__':
    main()
