import os
import re
import shutil
from tqdm import tqdm

def is_sentence_fragment(prev_line, curr_line):
    """
    This function tries to decide if curr_line is a continuation of a sentence, or the start of a new sentence.
    """
    if curr_line and curr_line[0].islower() and prev_line and prev_line[-1] not in '.!?':
        return True
    if curr_line and not curr_line[0].isupper() and not re.match(r'^\d+\.|-', curr_line):
        return True
    if re.match(r'''
        ^           # Sor eleje
        [0-9\.,:;!?]+  # Egy vagy több számjegy vagy központozási jel
        $           # Sor vége
        ''', curr_line, re.VERBOSE):
        return True
    if re.match(r'''
        ^           # Sor eleje
        [-.,:;!?]+  # Egy vagy több kötőjel vagy központozási jel
        $           # Sor vége
        ''', curr_line, re.VERBOSE):
        return True
    if re.match(r'''
        ^\(.*$      # Nyitott zárójel a sor elején
        ''', curr_line, re.VERBOSE) or re.match(r'''
        ^.*\)$      # Zárt zárójel a sor végén
        ''', curr_line, re.VERBOSE):
        return True
    if re.match(r'''
        \d{4}\.     # Négy számjegy (év) és egy pont
        \s+         # Egy vagy több szóköz
        \w+         # Egy vagy több alfanumerikus karakter (hónap)
        \s+         # Egy vagy több szóköz
        \d{1,2}\.   # Egy vagy két számjegy (nap) és egy pont
        $           # Sor vége
        ''', curr_line, re.VERBOSE):
        return True
    if re.match(r'''
        ^\d+\.$     # Számokból és pontból álló sor
        ''', curr_line, re.VERBOSE) or re.match(r'''
        ^[a-zA-Z]\)$ # Betűből és záró zárójelből álló sor
        ''', curr_line, re.VERBOSE):
        return True
    return False


def is_too_short(line):
    """
    Ez a függvény megpróbálja eldönteni, hogy egy sor túl rövid-e ahhoz, hogy teljes mondat legyen.
    """
    words = line.split()
    if len(words) < 3:
        return True
    return False


def clean_and_correct_sentences(file_path):
    """
    Ez a függvény beolvassa a megadott txt fájlt, eltávolítja az üres sorokat és
    összeilleszti a hibásan kettévágott mondatokat. Visszaadja az eredeti és a javított tartalmat.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    corrected_sentences = []
    temp_sentence = ''
    modifications = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        original_lines = [line]
        while is_too_short(line) or (temp_sentence and is_sentence_fragment(temp_sentence, line)):
            if temp_sentence:
                temp_sentence += ' ' + line
            else:
                temp_sentence = line
            i += 1
            if i >= len(lines):
                break
            line = lines[i].strip()
            if line:
                original_lines.append(line)

        if temp_sentence:
            if original_lines and len(original_lines) > 1:
                modifications.append((original_lines, temp_sentence))
            corrected_sentences.append(temp_sentence)
            temp_sentence = line
        else:
            temp_sentence = line

        i += 1

    if temp_sentence:
        corrected_sentences.append(temp_sentence)

    original_content = ''.join(lines)
    corrected_content = '\n'.join(corrected_sentences) + '\n'

    return original_content, corrected_content, modifications


class FileChecker(object):

    def CheckFolder(self, input_folder_path, output_folder_path, log_file_path) -> None:
        """
        Ez a függvény bejárja a megadott bemeneti mappát, minden txt fájlon meghívja a
        clean_and_correct_sentences függvényt, és az eredményeket a kimeneti mappába menti.
        """
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for filename in tqdm(os.listdir(input_folder_path)):
            if filename.endswith('.txt'):
                input_file_path = os.path.join(input_folder_path, filename)
                output_file_path = os.path.join(output_folder_path, filename)

                original_content, corrected_content, modifications = clean_and_correct_sentences(input_file_path)

                if modifications:
                    with open(output_file_path, 'w', encoding='utf-8') as file:
                        file.write(corrected_content)

                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"File: {input_file_path}\n")
                        for original, corrected in modifications:
                            log_file.write(f"Original: {' | '.join(original)}\n")
                            log_file.write(f"Corrected: {corrected}\n\n")
                else:
                    shutil.copy(input_file_path, output_file_path)
