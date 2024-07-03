import os
import uroman
import pandas as pd


def create_transliteration_data(file_path):
    directory, filename = os.path.split(file_path)
    print(filename)

    assert os.path.isfile(file_path), f"Input file path {file_path} not found"
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.read().splitlines()
    transliterations = uroman.romanize(sentences, './temp')

    print(f"Number of sentences: {len(sentences)}")
    print(f"Number of transliterations: {len(transliterations)}")
    assert len(sentences) == len(transliterations)
    # write transliterations:
    data_df = pd.DataFrame({'text': sentences, 'transliteration': transliterations})

    data_df.to_csv(f"{directory}/text_transliterations_{filename}.csv", index=False, escapechar='\\')
    print(f"Saved in {directory}/text_transliterations_{filename}.csv")


create_transliteration_data("/mounts/data/proj/orxhelili/ten_percent_languages/group2.txt")
create_transliteration_data("/mounts/data/proj/orxhelili/ten_percent_languages/group1.txt")
