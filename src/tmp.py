# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""""
    with tf.Graph().as_default():
            bert_module = hub.Module(BERT_MODEL_HUB)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                        tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case
            )

tokenizer = create_tokenizer_from_hub_module()
