import torch
from torch import nn
from transformers import BertModel, GPT2Model, LlamaModel, LlamaConfig, T5Model

class MergedTransformer(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', gpt_model_name='gpt2', llama_model_name='meta-llama/Llama-2-7b', t5_model_name='t5-base'):
        super(MergedTransformer, self).__init__()

        # Initialize models
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.gpt = GPT2Model.from_pretrained(gpt_model_name)
        llama_config = LlamaConfig.from_pretrained(llama_model_name)
        self.llama = LlamaModel(llama_config)
        self.t5 = T5Model.from_pretrained(t5_model_name)

        # Projection layers
        hidden_size = 768
        self.bert_projection = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.gpt_projection = nn.Linear(self.gpt.config.n_embd, hidden_size)
        self.llama_projection = nn.Linear(self.llama.config.hidden_size, hidden_size)
        self.t5_projection = nn.Linear(self.t5.config.d_model, hidden_size)

        # Attention mechanism for model fusion
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)

        # Final layers
        self.fusion_layer = nn.Linear(hidden_size * 4, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        # Forward passes
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        llama_output = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        t5_output = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids)

        # Project outputs
        bert_hidden = self.bert_projection(bert_output.last_hidden_state)
        gpt_hidden = self.gpt_projection(gpt_output.last_hidden_state)
        llama_hidden = self.llama_projection(llama_output.last_hidden_state)
        t5_hidden = self.t5_projection(t5_output.last_hidden_state)

        # Combine outputs using attention mechanism
        combined_hidden = torch.stack([bert_hidden, gpt_hidden, llama_hidden, t5_hidden], dim=0)
        attended_output, _ = self.attention(combined_hidden, combined_hidden, combined_hidden)

        # Fusion and final output
        fused_output = self.fusion_layer(torch.cat([bert_hidden, gpt_hidden, llama_hidden, t5_hidden], dim=-1))
        output = self.output_layer(fused_output + attended_output[-1])

        return output

    def fine_tune(self, task='classification', num_labels=2):
        if task == 'classification':
            self.task_head = nn.Linear(768, num_labels)
        elif task == 'generation':
            self.task_head = nn.Linear(768, self.gpt.config.vocab_size)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def task_specific_forward(self, input_ids, attention_mask, task='classification'):
        output = self.forward(input_ids, attention_mask)
        if task == 'classification':
            return self.task_head(output[:, 0, :])  # Use [CLS] token for classification
        elif task == 'generation':
            return self.task_head(output)  # Use all tokens for generation
        else:
            raise ValueError(f"Unsupported task: {task}")

def get_merged_transformer(backend='pytorch'):
    if backend == 'pytorch':
        return MergedTransformer()
    elif backend == 'jax':
        raise NotImplementedError("JAX backend not implemented yet")
    elif backend == 'tensorflow':
        raise NotImplementedError("TensorFlow backend not implemented yet")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
