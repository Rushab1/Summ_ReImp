import os
import nltk
from tqdm import tqdm

dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'

# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]

#split article into sentences at <t> </t>
def extract_sentences(text):
    text = text.replace("</t>", "")
    return text.split("<t>")

def create_nyt_story_file(articles_file, abstracts_file, save_dir, url_save_file, use_nltk = False):
    f = open(articles_file).read().strip()
    f = f.replace("\n\n", "\n<unknown>\n").split("\n")

    a = open(abstracts_file).read().strip()
    a = a.replace("\n\n", "\n<unknown>\n").split("\n")

    url_file = open(url_save_file, "w")

    #sent_tokenize the article and abstracts
    for i in tqdm(range(0, len(f))):
        try:
            f[i] = f[i].decode("utf-8")
            a[i] = a[i].decode("utf-8")
        except:
            pass
        art_sent = nltk.sent_tokenize(f[i])
        abs_sent = nltk.sent_tokenize(a[i])
        abs_sent = ["@highlight \n\n" + sent for sent in abs_sent]

        output = "\n\n".join(art_sent)
        output += "\n\n" + "\n\n".join(abs_sent)

        g = open(os.path.join(save_dir, str(i) + ".story"), "w")
        g.write(str(output))
        g.close()

        url_file.write(str(i) + ".story\n")

    url_file.close()

def convert_nyt_to_story(nyt_file, abstracts_file, save_dir, url_file):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    create_nyt_story_file(nyt_file, abstracts_file, save_dir, url_file)

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--articles_file", type=str, required = True)
    args.add_argument("--abstracts_file", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True )
    args.add_argument("--dataset_type", type=str, required=True ) #test, train or valid
    opts = args.parse_args()

    pjoin = os.path.join
    output_dir = os.path.abspath(opts.output_dir)

    os.system("rm -rf " + output_dir)
    os.mkdir(output_dir)

    RAW_PATH        = pjoin(output_dir, "conversion_output")
    TOKENIZED_PATH  = pjoin( output_dir, "tokenized")
    JSON_PATH       = pjoin(output_dir, "finished_files")
    MAP_PATH        = pjoin(output_dir, "maps")
    BERT_DATA_PATH  = pjoin( output_dir, "Bert_Data")
    OUTPUT_PATH     = pjoin(output_dir, "output")

    os.mkdir(TOKENIZED_PATH)
    os.mkdir(JSON_PATH)
    os.mkdir(MAP_PATH)
    os.mkdir(BERT_DATA_PATH)
    os.mkdir(OUTPUT_PATH)

    convert_nyt_to_story(   opts.articles_file,
                            opts.abstracts_file,
                            RAW_PATH,
                            pjoin( MAP_PATH, "mapping_" + opts.dataset_type + ".txt")
                         )

    classpath = os.path.abspath("../packages/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar")
    os.environ["CLASSPATH"] = classpath

    cmd = [ "python", "preprocess.py",
            "-mode", "tokenize",
            "-raw_path ", RAW_PATH,
            "-save_path", TOKENIZED_PATH,
            "-log_file", "log.log",
            "-dataset", opts.dataset_type
           ]
    os.system("  ".join(cmd))

    cmd = [     "python preprocess.py",
                "-mode format_to_lines",
                "-raw_path", TOKENIZED_PATH,
                "-save_path", pjoin(JSON_PATH, "mid"),
                "-n_cpus 15 -use_bert_basic_tokenizer false",
                "-map_path", MAP_PATH,
                "-log_file log.log",
                "-dataset", opts.dataset_type
            ]
    os.system("  ".join(cmd))

    # cmd = [
                # "python", "preprocess.py",
                # "-mode", "format_to_bert",
                # "-raw_path", JSON_PATH,
                # "-save_path", BERT_DATA_PATH,
                # "-lower",
                # "-n_cpus", "15",
                # "-log_file", "log.log",
                # "-dataset", opts.dataset_type
            # ]
    # os.system("  ".join(cmd))
