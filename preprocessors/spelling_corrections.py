from typing import Set, Union, List
import pandas as pd
import os


def find_similar_sentences(set1, set2, filename):
    def levenshtein_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    similar_sentences = []
    similar_sentencces_as_list = []

    for sentence1 in set1:
        for sentence2 in set2:
            distance = levenshtein_distance(sentence1, sentence2)
            if 0 < distance <= 3:
                similar_sentences.append((sentence1, sentence2, distance))
                similar_sentencces_as_list.append(sentence1)
                similar_sentencces_as_list.append(sentence2)

    df = pd.DataFrame(similar_sentences, columns=['original', 'rephrased', 'distance'])
    df['file'] = filename

    return df, similar_sentencces_as_list


def remove_list_elements_from_set(original_set, elements_to_remove):
    for element in elements_to_remove:
        original_set.discard(element)  # discard does not raise an error if the element is not present in the set
    return original_set