EMBEDDING_DIMEN = 100
RNN_DIM = 256
BATCH_SIZE = 100
MAX_SEQ_LEN = 160

LEARNING_RATE = 0.001
OPTIMIZER = "Adam"

TRAIN_FILE = "./data/data.txt"
# TEST_FILE = "./data/test.txt"
# VALID_FILE = "./data/valid.txt"
VOCAB_FILE = "./data/vocab.in"
DATA_FILE = "./data/data_raw.txt"
EMBEDDINGS_FILE = "./data/embeddings.txt"

nepochs = 10
top_threshold = 0.8
bottom_threshold = 0.5
