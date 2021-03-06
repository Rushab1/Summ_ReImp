import os
import nltk
import glob
import json
from tqdm import tqdm

dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'

pjoin = os.path.join

def create_nyt_story_file(articles_file, abstracts_file, save_dir, url_save_file, use_nltk = False):
    f = open(articles_file).read().strip()
    f = f.replace("\n\n", "\n<unknown>\n")
    f = f.split("\n")

    a = open(abstracts_file).read().strip()
    a = a.replace("\n\n", "\n<unknown>\n")
    a = a.replace("<t>", " ")
    a = a.replace("</t>", " ")
    a = a.split("\n")

    url_file = open(url_save_file, "w")

    #sent_tokenize the article and abstracts
    for i in tqdm(range(0, len(f))):
        try:
            f[i] = f[i].decode("utf-8")
            a[i] = a[i].decode("utf-8")
            print(a[i])
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

def create_individual_processed_files(dataset, dataset_types):
    for dataset_type in dataset_types:
        cnt = 0
        save_dir = pjoin("../Data/cnndm/Processed/", dataset_type)
        if not os.path.exists( save_dir ):
            os.mkdir(save_dir)

        finished_files_dir = pjoin( "../Data",
                                    dataset,
                                    "Processed/finished_files")

        file_list = glob.glob(pjoin(finished_files_dir,
                                    "mid.%s.[0-9]*.json" % dataset_type ))
        file_list = sorted(file_list)
        print(file_list)
        for fname in tqdm(file_list):
            f = open(fname)
            json_files = json.load(f)
            f.close()

            for data in json_files:
                with open(pjoin(save_dir, "%d.json"%cnt), "w") as save_file:
                    json.dump(data, save_file)
                    save_file.close()
                cnt += 1


def preprocess(opts):
    dataset             = opts.dataset
    output_dir          = pjoin("../Data", dataset, "Processed")
    output_dir          = os.path.abspath(output_dir)

    os.system("rm -rf " + output_dir)
    os.mkdir(output_dir)

    JSON_PATH           = pjoin(output_dir, "finished_files")
    MAP_PATH            = pjoin(output_dir, "maps")
    os.mkdir(JSON_PATH)
    os.mkdir(MAP_PATH)

    for dataset_type in opts.dataset_types:
        articles_file   = pjoin("../Data/", dataset, "Raw", "%s.txt.src" % dataset_type)
        abstracts_file  = pjoin("../Data/", dataset, "Raw", "%s.txt.tgt" % dataset_type)

        RAW_PATH        = pjoin(output_dir, "conversion_output_%s" % dataset_type)
        TOKENIZED_PATH  = pjoin( output_dir, "tokenized_%s" % dataset_type)

        os.mkdir(TOKENIZED_PATH)

        convert_nyt_to_story(   articles_file,
                                abstracts_file,
                                RAW_PATH,
                                pjoin( MAP_PATH, "mapping_" + dataset_type + ".txt")
                            )

        classpath = os.path.abspath("../packages/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar")
        os.environ["CLASSPATH"] = classpath

        cmd = [ "python", "preprocess.py",
                "-mode", "tokenize",
                "-raw_path ", RAW_PATH,
                "-save_path", TOKENIZED_PATH,
                "-log_file", "log.log",
                "-dataset", dataset_type
                ]
        os.system("  ".join(cmd))

        cmd = [     "python preprocess.py",
                    "-mode format_to_lines",
                    "-raw_path", TOKENIZED_PATH,
                    "-save_path", pjoin(JSON_PATH, "mid"),
                    "-n_cpus 15 -use_bert_basic_tokenizer false",
                    "-map_path", MAP_PATH,
                    "-log_file log.log",
                    "-dataset", dataset_type
                    ]
        os.system("  ".join(cmd))

    create_individual_processed_files(opts.dataset, opts.dataset_types)

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, required=True)
    # args.add_argument("--articles_file", type=str, required = True)
    # args.add_argument("--abstracts_file", type=str, required=True)
    # args.add_argument("--output_dir", type=str, required=True )
    args.add_argument("--dataset_types", nargs='+', type=str, required=True ) #test, train or valid, all
    opts                = args.parse_args()

    preprocess(opts)
