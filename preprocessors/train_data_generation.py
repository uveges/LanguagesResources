import os
import re
import string
from typing import List

import docx2txt
# import hu_core_news_trf
import pandas as pd
from importlib_resources import files
from tqdm import tqdm

import config
from preprocessors.sentence_splitter import SentenceSplitter
from src.wordfiletriplet import WordFileTriplet, NameException, ContentAexception, ContentBexception, ContentCexception


def write_file(result: str, new_name: str) -> None:
    fajl = open(new_name, "w+", encoding="utf8")
    fajl.write(result)


class Generator:

    def __init__(self,
                 mode: str,
                 name_original: str,
                 name_rephrased: str,
                 name_full: str,
                 corpora_folder: str,
                 txt_folder: str,
                 sentence_segmented_folder: str):

        self.begin_sentence_pat = re.compile(
            r"^(?:[a-zöüóőúűéáí]|\d+[\s.]{,3}§|[A-ZÍŰÁÉÚŐÓÜÖ][a-zíűáéúőóüö]?\.\s?(?:,|és|[A-ZÍŰÁÉÚŐÓÜÖ][a-zíűáéúőóüö]?\s?\.)|\(\s?[XVI]+\s?\.?\s?\d{1,2}\s?\.?\s?\)|E?BH\d{4}\.|\d+\s?(?:[-–]\s?\d+\s?|/[A-Z])?\.?\s?(?:§|[cC]ikk))")
        self.ending_sentence_pat = re.compile(
            r"(?:\(\s?[XVI]+\s?\.?\s?\d{1,2}\s?\.?\s?\)|\.{2,4}\)?|…{1,2}\.{,2}|[A-ZÍŰÁÉÚŐÓÜÖ][a-zíűáéúőóüö]?\s?\.|E?BH\d{4}\.|[A-ZÍŰÁÉÚŐÓÜÖ][a-zíűáéúőóüö]{,5}?tv\.)$")
        self.table_of_contents = re.compile(r"^\t*([0-9][\.]\s?)+[\t]?.+[0-9]+\.?(\r\n|\r|\n)?$")
        self.urls = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        self.ALREADY_FOUND_PREFIXES = []
        self.mode = mode
        self.sentence_splitter = SentenceSplitter(language='hu', non_breaking_prefix_file=str(files("resources") / 'hu.txt'))
        self.name_original = name_original
        self.name_rephrased = name_rephrased
        self.name_full = name_full
        self.corpora_folder = corpora_folder
        self.txt_folder = txt_folder
        self.sentence_segmented_folder = sentence_segmented_folder

    def docxtotxt(self, input_directory_in_str: str) -> None:
        """
        Converting the .doc, .docx files into a .txt format.

        :param input_directory_in_str: Folder containing the MS Word files
        :param output_directory_in_str: Folder to store the txt files
        :return: None
        """

        non_processable_files = []
        print("\r\nConverting MS Word files:")

        for file in tqdm(os.listdir(input_directory_in_str)):
            filename = os.fsdecode(file)
            if filename.endswith(".docx"):
                input_full_path = os.path.join(input_directory_in_str, filename)
                output_full_path = os.path.join(self.txt_folder, filename.replace(".docx", ".txt"))
                try:
                    content = docx2txt.process(input_full_path)
                    write_file(content, output_full_path)
                except Exception as e:
                    print(".docx hiba: " + input_full_path)
                    print(e)
            elif filename.endswith(".doc"):
                input_full_path = os.path.join(input_directory_in_str, filename)
                output_full_path = os.path.join(self.txt_folder, filename.replace(".doc", ".txt"))
                try:
                    os.system(f'antiword -f {input_full_path} > {output_full_path}')
                except Exception as e:
                    print(".doc hiba: " + input_full_path)
                    print(e)
            else:
                non_processable_files.append(filename)

        if non_processable_files:
            print(
                "Non-processed files in directory (please check for errors manually, e.g. unconventional file names):")
            print(non_processable_files)
        else:
            print("\r\nAll files successfully processed!")

    def create_training_dataset(self):

        path_for_original = os.path.join(self.corpora_folder, self.name_original)
        path_for_rephrased = os.path.join(self.corpora_folder, self.name_rephrased)

        original_sentences = pd.read_excel(path_for_original)
        rephreased_sentences = pd.read_excel(path_for_rephrased)

        original_sentences['Label'] = "original"
        rephreased_sentences['Label'] = "rephrased"

        concatenated = pd.concat([original_sentences, rephreased_sentences], axis=0)
        concatenated.to_excel(os.path.join(self.corpora_folder, self.name_full), index=False)

    def segment_txts_to_sentences(self, debug: bool = False) -> None:
        """
        Segmenting txt files into sentences. Output: .sent files, where one line is one sentence.

        :param debug:
        :return:
        """

        print("Segmenting files to sentences: ")
        unprocessable = []
        filtered = []

        for file in tqdm(os.listdir(self.txt_folder)):
            filename = os.fsdecode(file)

            if filename.endswith(".txt"):

                full_path = os.path.join(self.txt_folder, file)
                with open(full_path, 'r', encoding="utf8") as txt:
                    content = txt.read()

                content = re.sub("\\r\\n", " ", content)
                content = re.sub("\\r", " ", content)
                content = re.sub("\\n", " ", content)
                sentences = self.sentence_splitter.split(content)
                sentences = [sentence for sentence in sentences if sentence != ""]  # remove empty lines
                sentences = self.filter_out_noise_from_data(sentences, debug, filtered)

                to_write = "\r\n".join(sentences)
                try:
                    file_to_write = os.path.join(self.sentence_segmented_folder, filename.replace(".txt", config.TEMP_FILE_EXTENSIONS['sentence_segmented']))
                    result = open(file_to_write, "w+", encoding="utf8")
                    result.write(to_write)
                    result.close()
                except Exception as e:
                    print(e)
            else:
                unprocessable.append(filename)

        if unprocessable:
            print("Files unable to segment: ")
            for u in unprocessable:
                print(u)
        else:
            print("\r\nAll files successfully segmented!")

        if debug:
            print("Filtered sentences:")
            print(filtered)

    def filter_out_noise_from_data(self, sentences: List, debug: bool, filtered: List) -> List:
        """
        Correcting errors made by the sentence splitter, as well as reduce other noise in the data, like throwing out
        headings, table of contents etc.

        :param sentences: Unfiltered list of segmented sentences
        :param debug: If printing the sentences considered as noise is required
        :param filtered: If debug=True, list for storing the thrown sentences
        :return:
        """

        sentences = self.filter_out_noise(sentences)

        if config.FILTERING and debug:
            sentences = self.filter_out_other(sentences, debug, filtered)
        elif config.FILTERING:
            sentences = self.filter_out_other(sentences, debug)
        return sentences

    def filter_out_other(self, sentences: List, debug: bool, filtered=None) -> List:
        """
        Based on settings in config.py, filters "headings_and_table_of_contents" and "URLs" from the previously segmented
        sentences.

        :param sentences: List of sentences
        :param debug: If true, the function saves the sentences that were dropped by filtering
        :param filtered: optional: if debug=True, the function needs an empty list as argument to return dropped sentences
         during filtering
        :return:
        """

        tmp = []

        if config.FILTER["headings_and_table_of_contents"]:
            for sent in sentences:
                matched = re.match(self.table_of_contents, sent)
                if not matched:
                    tmp.append(sent)
                else:
                    if debug:
                        filtered.append(sent)
        if config.FILTER["URLs"]:
            if len(tmp) > 0:
                for index, sent in enumerate(tmp):
                    matched = self.find_url(sent)
                    if matched:
                        for x in matched:
                            tmp[index] = tmp[index].replace(x, "")
                        if debug:
                            for x in matched:
                                filtered.append(x)
            else:
                for index, sent in enumerate(sentences):
                    matched = self.find_url(sent)
                    if matched:
                        for x in matched:
                            tmp.append(sentences[index].replace(x, ""))
                        if debug:
                            for x in matched:
                                filtered.append(x)

        if config.FILTER["URLs"] or config.FILTER["headings_and_table_of_contents"]:
            return tmp
        else:
            return sentences

    def find_url(self, sentence: str) -> List:
        url = re.findall(self.urls, sentence)
        return [x[0] for x in url]

    def filter_out_noise(self, sentences: List) -> List:
        """
        A common problem with sentence splitter is the creation of "fake" sentences, e.g. beginning with a lowercase letter.
        This function concatenates these sentence fragments into "real" ones.

        :param sentences: List of un-checked sentences
        :return: Noise-reduced list of sentences
        """

        results = sentences
        results2 = []

        i = 0
        skip = 0

        while i < len(results):
            sentence = results[i]
            j = i + 1
            while j < len(results):
                if self.begin_sentence_pat.search(results[j]) or self.ending_sentence_pat.search(sentence):
                    sentence += " "
                    sentence += results[j]
                    skip += 1
                    j += 1
                elif skip == 0:
                    j += 1
                    break
                else:
                    i += skip
                    skip = 0
                    break
            i += 1
            results2.append(sentence)

        return results2

    def gettriplets(self, filenames: list) -> List:
        word_document_triplets = []

        print("\r\nGenerating corpora:\r\n")

        # bejárjuk a már ismert fájlneveket
        for filename in tqdm(filenames):

            # print(f'Processing: {filename}')
            ending = config.TEMP_FILE_EXTENSIONS['sentence_segmented']
            to_search = '([A-Z][0-9]+)([A-Z])(' + ending + ')'  # KFO original
            # to_search = '([0-9]+)([A-Z])(' + ending + ')'             # KFO new
            match = re.search(to_search, filename)

            name = match.group(1)  # pl. A1
            version = match.group(2)  # melyik változat: A, B vagy C
            extension = match.group(3)

            # meg kell keresnünk a másik két verziót, beolvasni a tartalmukat, és az eredményeket eltárolni!
            mostvizsgalt = os.path.join(self.sentence_segmented_folder, filename)  # ez az, amit megtaláltunk, kell a másik kettő

            other1 = ""
            other2 = ""
            if version == 'A':  # ha az A-t találtuk meg, kell a B és C
                other1 = name + "B" + extension
                other2 = name + "C" + extension
            if version == 'B':  # ha az B-t találtuk meg, kell a A és C
                other1 = name + "A" + extension
                other2 = name + "C" + extension
            if version == 'C':  # ha az C-t találtuk meg, kell a A és B
                other1 = name + "A" + extension
                other2 = name + "B" + extension

            other1full = os.path.join(self.sentence_segmented_folder, other1)
            other2full = os.path.join(self.sentence_segmented_folder, other2)

            # ha már feldolgoztuk ezt a fájlt, haladjunk tovább
            if name in self.ALREADY_FOUND_PREFIXES:
                continue

            # ha még nem, dolgozzuk fel
            self.ALREADY_FOUND_PREFIXES.append(name)

            document_triplet = WordFileTriplet()  # objektum, ami tárolja a fenti 3 fájl tartalmát

            for index, fajl in enumerate([mostvizsgalt, other1full, other2full]):

                with open(fajl, 'r', encoding='UTF8') as f:
                    sentences = f.readlines()

                    try:
                        match = re.search(to_search, fajl)
                        version = match.group(2)  # nem tudjuk, melyik verzió van éppen meg, ezért lekérdezzük

                        if version == 'A':  # ha ez éppen az A verzió, akkor a ContentA -ba tároljuk el
                            if index == 0:  # ha még nem állítottuk be a tároló objektum nevét, akkor megtesszük
                                document_triplet.name = name
                            document_triplet._content_A = sentences

                        if version == 'B':
                            if index == 0:
                                document_triplet.name = name
                            document_triplet._content_B = sentences

                        if version == 'C':
                            if index == 0:
                                document_triplet.name = name
                            document_triplet._content_C = sentences

                    except NameException:
                        print("The object has its name already set!")
                    except ContentAexception:
                        print("This object has its ContentA attribute already set!")
                    except ContentBexception:
                        print("This object has its ContentB attribute already set!")
                    except ContentCexception:
                        print("This object has its ContentC attribute already set!")

            word_document_triplets.append(document_triplet)

        return word_document_triplets

    def get_segmented_filenames(self) -> List:
        filenames = []

        for file in os.listdir(self.sentence_segmented_folder):
            filename = os.fsdecode(file)
            if filename.endswith(config.TEMP_FILE_EXTENSIONS['sentence_segmented']):
                filenames.append(filename)
        return filenames

    def covert_to_excels(self, lista: list, filename: str) -> None:
        # Az input itt listák listája, ezt alakítjuk egyetlen listává
        flat_list = [item for sublist in lista for item in sublist]

        if config.FILTER["minimum_characters_for_a_sentence"]:
            flat_list = [x for x in flat_list if len(x) >= config.FILTER["minimum_characters_for_a_sentence"]]
        if config.FILTER["punctuations"]:
            flat_list = [x.translate(str.maketrans('', '', string.punctuation)) for x in flat_list]

        print(f"Length of corpus: {len(flat_list)}")

        df = pd.DataFrame(flat_list, columns=['Text'])
        df.to_excel(os.path.join(self.corpora_folder, filename), index=False)
