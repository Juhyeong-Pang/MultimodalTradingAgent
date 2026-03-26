from dataclasses import dataclass

@dataclass
class Config:
    MAX_LEN = 16
    BATCH_SIZE = 16
    LR = 2e-5
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 4
    FF_DIM = 512
    NUM_LAYERS = 2

config = Config()