import re
import nltk
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize




def custom_split(text):

    unwanted_words_list = [
        "Title:",
        "presentation:",
        "patient:",
        "Age:",
        "Gender:",
        "study_findings:",
        "discussion:",
        # "\n",
    ]

    blocks = re.split(r"\n{3,}", text) # block that are split by two or more consecutive newlines

    results = [] # all sentences
    for block in blocks:
        pieces = re.split(r"([.!?])", block)

        combined = [] # all sentences in a block
        for i in range(0, len(pieces), 2):
            # `pieces[i]` is text (might be empty/whitespace if there's punctuation in a row)
            chunk = pieces[i].strip() # single sentence
            
            # If there's a next element (the punctuation), attach it
            if i + 1 < len(pieces):
                chunk += pieces[i + 1]
            chunk = chunk.strip()

            for word in unwanted_words_list:
                    chunk = chunk.replace(word, "")
            chunk = chunk.strip()
            

            if chunk: # process one sentence, chunk is a string

                word_counts = len(re.split(r"[ \n]+", chunk))

                if word_counts > 30 and "\n" in chunk:
                    subchunks = re.split(r"\n", chunk)
                    for subchunk in subchunks:
                        if subchunk:
                            subchunk = subchunk.replace("\xa0", " ")
                            combined.append(subchunk)
                else:
                    chunk = chunk.replace("\xa0", " ")
                    chunk = chunk.replace("\n", " ")
                    combined.append(chunk)


        if combined:
            results.extend(combined)

    return results

def word_count_stats(list_of_strings):
    """
    Given a list of text strings, returns:
      - The maximum word count
      - The minimum word count
      - The mean (average) word count
    """
    # 1. Split each string by whitespace and count the words
    counts = [len(s.split()) for s in list_of_strings]

    # 2. Compute max, min, and mean of the counts
    max_count = max(counts)
    min_count = min(counts)
    mean_count = sum(counts) / len(counts) if len(counts) > 0 else 0
    total_count = sum(counts)

    median_count = sorted(counts)[len(counts)//2]

    return max_count, min_count, mean_count, median_count, total_count


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--txt_path", type=str)

    args = parser.parse_args()

    return args


def main():
    # nltk.download('punkt')

    ### if a chunk is too long, we split chunk by "\n"

    args = get_args()

    with open(args.txt_path, "r") as f:
        text = f.read()
    print(text)
    splitted_list = custom_split(text)
    # return
    print(splitted_list)

    num_sentences = len(splitted_list)
    print(num_sentences)

    max_len_words, min_len_words, mean_len_words, median_count, total_count = word_count_stats(splitted_list)
    print(max_len_words, min_len_words, mean_len_words, median_count, total_count)

    # sentences_list = sent_tokenize(text)
    # print(sentences_list)




if __name__ == "__main__":

    main()