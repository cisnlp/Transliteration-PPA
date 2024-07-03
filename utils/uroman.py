import codecs
import concurrent.futures
import os.path
import os

# Path to the uroman executable
uroman_path = "/mounts/Users/cisintern/orxhelili/uroman/bin/"


def romanize_chunk(sentences_chunk, chunk_id, temp_path, lang=None):
    in_path = f"{temp_path}/sentence_{chunk_id}.txt"
    out_path = f"{temp_path}/sentence_roman_{chunk_id}.txt"

    with codecs.open(in_path, "w", "utf-8") as fa_file:
        fa_file.write("\n".join(sentences_chunk))

    if lang is None:
        os.system(uroman_path + "uroman.pl < {0} > {1} ".format(in_path, out_path))
    else:
        os.system(uroman_path + "uroman.pl -l {0} < {1} > {2} ".format(lang, in_path, out_path))

    with open(out_path, "r", encoding='utf-8') as f1:
        romanized = [line.strip() for line in f1]

    # Clean up
    os.remove(in_path)
    os.remove(out_path)

    return romanized


def romanize(sentences: list[str], temp_path='./temp', lang=None, num_workers=16):
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    chunk_size = len(sentences) // num_workers + 1
    sentence_chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

    romanized_sentences = [None] * len(sentence_chunks)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {executor.submit(romanize_chunk, chunk, i, temp_path, lang): i for i, chunk in enumerate(sentence_chunks)}

        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                romanized_sentences[chunk_id] = future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # Flatten the list of lists while preserving order
    romanized_sentences_flattened = [sentence for chunk in romanized_sentences for sentence in chunk]

    return romanized_sentences_flattened
