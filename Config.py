class Config():

    HUBERT = 'facebook/hubert-base-ls960'
    WAV2VEC2 = 'facebook/wav2vec2-base-960h'
    mHuBERT = 'utter-project/mHuBERT-147'
    MGTEBASE = 'Alibaba-NLP/gte-multilingual-base'
    WHISPER = "openai/whisper-large-v3-turbo"

    def __init__(self):
        return

    def get_subnet_insize(self):
        if self.transformer_checkpoint == self.HUBERT:
            return 768
        elif self.transformer_checkpoint == self.WAV2VEC2:
            return 768
