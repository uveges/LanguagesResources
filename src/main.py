import os
from preprocessors.train_data_generation import Generator
from importlib_resources import files
from preprocessors.segmentation_checker import FileChecker
import pandas as pd

CORPORA_FOLDER = str(files("resources") / "excel_corpora")
TXT_FOLDER = str(files("resources") / "txts")
TXT_CORRECTED_FOLDER = str(files("resources") / "txts_corrected")
CORRECTION_LOGFILE = str(files("resources") / "txt_correction_log.txt")
SENTENCE_SEGMENTED_FOLDER = str(files("resources") / "segmented")
WORD_FILES_FOLDER = str(files("resources") / "KFO")


def main() -> None:
    """
    Generating corpus from MS Word files.
    :return:
    :rtype:
    """

    # docx to txt, sentence segmentation, segmentation corrections
    # phase_one()
    # creating excel corpora from segmented, corrected txts
    phase_two()


def phase_two():

    # original_sentences, rephrased_sentences = ([] for i in range(2))
    #
    # generator = Generator(mode="splitter",
    #                       name_original="original.xlsx",
    #                       name_rephrased="rephrased.xlsx",
    #                       name_full="full_dataset.xlsx",
    #                       corpora_folder=CORPORA_FOLDER,
    #                       txt_folder=TXT_CORRECTED_FOLDER,
    #                       sentence_segmented_folder=SENTENCE_SEGMENTED_FOLDER
    #                       )
    #
    # filenames = generator.get_segmented_filenames()
    # doc_triplets = generator.gettriplets(filenames=filenames)
    #
    # for doc_triplet in doc_triplets:
    #     original_sentences.append(doc_triplet.get_subcorpora()[0])
    #     rephrased_sentences.append(doc_triplet.get_subcorpora()[1])
    #
    # generator.covert_to_excels(original_sentences, filename="original.xlsx")
    # generator.covert_to_excels(rephrased_sentences, filename="rephrased.xlsx")
    # generator.create_training_dataset()

    full_corpus = str(files("resources") / "excel_corpora" / "full_dataset.xlsx")
    deduplicated_coprus = str(files("resources") / "excel_corpora" / "FINAL_full_dataset.xlsx")
    remove_duplicates_from_excel(original_path=full_corpus,
                                 result_path=deduplicated_coprus)


def phase_one():
    """
    Initialize technical folders.
    Converting .docx files to txts.
    Sentence segmentation of each txt.
    Correction of the segmented files into a new folder.
    :return:
    :rtype:
    """


    initialize_folders([TXT_FOLDER, SENTENCE_SEGMENTED_FOLDER, CORPORA_FOLDER])

    generator = Generator(mode="splitter",
                          name_original="original.xlsx",                        # not used here
                          name_rephrased="rephrased.xlsx",                      # not used here
                          name_full="full_dataset.xlsx",                        # not used here
                          corpora_folder=CORPORA_FOLDER,                        # not used here
                          txt_folder=TXT_FOLDER,
                          sentence_segmented_folder=SENTENCE_SEGMENTED_FOLDER   # not used here
                          )

    generator.docxtotxt(WORD_FILES_FOLDER)
    generator.segment_txts_to_sentences(debug=False)

    filechecker = FileChecker()
    filechecker.CheckFolder(input_folder_path=TXT_FOLDER,
                            output_folder_path=TXT_CORRECTED_FOLDER,
                            log_file_path=CORRECTION_LOGFILE)

def initialize_folders(folders: list) -> None:
    created = False
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created: {folder}")
            created = True
    if not created:
        print("All technical folder were already existed.")


def remove_duplicates_from_excel(original_path: str, result_path: str) -> None:
    """
    Eltávolítja a duplikált sorokat egy Excel fájlból a "Text" oszlop alapján.
    Removes potential duplicates (e.g. extreme short sentences, that may appear more than once by accident) based on the
    'Text' column.
    """
    df = pd.read_excel(original_path)
    df_cleaned = df.drop_duplicates(subset=['Text'], keep='first')
    df_cleaned.to_excel(result_path, index=False)


if __name__ == '__main__':
    main()
