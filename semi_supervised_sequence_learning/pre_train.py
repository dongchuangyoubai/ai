import os
import argparse
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia

import nltk
nltk.download('punkt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="auto_encoder", help="auto_encoder | language_model")
    args = parser.parse_args()

    if not os.path.exists("dbpedia_csv"):
        print("Downloading dbpedia dataset...")
        download_dbpedia()

    print("\nBuilding dictionary..")
    word_dict = build_word_dict()