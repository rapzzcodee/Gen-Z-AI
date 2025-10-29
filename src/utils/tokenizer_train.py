import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    '--input=data/processed/genz_dataset.jsonl --model_prefix=data/vocab/genz '
    '--vocab_size=8000 --model_type=bpe'
)

print("Tokenizer DONE!")
