import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentencepiece import SentencePieceProcessor

class GenerativeAIModel(nn.Module):
    def __init__(self, model_name="t5-base", sp_model_path="sp_model.model"):
        super(GenerativeAIModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.sp_processor = SentencePieceProcessor(model_file=sp_model_path)

    def forward(self, input_text):
        sp_tokens = self.sp_processor.encode(input_text, out_type=str)
        inputs = self.tokenizer(sp_tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return self.sp_processor.decode(self.sp_processor.encode(decoded_outputs))

    def generate(self, input_text, max_length=50):
        sp_tokens = self.sp_processor.encode(input_text, out_type=str)
        inputs = self.tokenizer(sp_tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs, max_length=max_length)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return self.sp_processor.decode(self.sp_processor.encode(decoded_outputs))

def load_model(model_path=None, sp_model_path="sp_model.model"):
    model = GenerativeAIModel(sp_model_path=sp_model_path)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    return model

# Example usage (commented out)
# model = load_model()
# input_text = "Translate the following English text to French: 'Hello, how are you?'"
# output = model.generate(input_text)
# print(output)
