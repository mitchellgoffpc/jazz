import torch
from model import GPT, GPTConfig

if __name__ == '__main__':
  from transformers import GPT2LMHeadModel, GPT2Tokenizer
  hfmodel = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  state_dict = hfmodel.state_dict()

  # Remap names
  replacements = {
    'transformer.': '',
    'h.': 'blocks.',
    'wte.': 'embed_tokens.',
    'wpe.': 'embed_pos.',
    'attn.c_attn': 'attn.qkv',
    'attn.c_proj': 'attn.proj',
    'mlp.c_fc': 'mlp.fc1',
    'mlp.c_proj': 'mlp.fc2',
    'ln_1.': 'ln1.',
    'ln_2.': 'ln2.',
    'ln_f.': 'ln.',
    'lm_head.': 'fc_out.',
  }
  linears = ['attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight']
  biases = ['attn.bias', 'attn.masked_bias']

  for src,dst in replacements.items():
    state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
  state_dict = {k:v for k,v in state_dict.items() if not any(x in k for x in biases)}
  state_dict = {k: v.transpose(-1, -2) if any(x in k for x in linears) else v for k,v in state_dict.items()}

  config = GPTConfig
  model = GPT(config)
  model.load_state_dict(state_dict)

  text = "What is the capital of France?"
  encoded_input = tokenizer(text, return_tensors='pt').input_ids

  logits = model(encoded_input)
  hf_logits = hfmodel(encoded_input).logits
  torch.testing.assert_close(logits, hf_logits, atol=0, rtol=2e-3)

  # tokens = encoded_input
  # for _ in range(5):
  #   tokens = torch.cat([tokens, model.sample(tokens)], dim=-1)
  # tokens = model.generate(encoded_input, max_new_tokens=5)
  # hf_tokens = hfmodel.generate(encoded_input, max_new_tokens=5)
  # print(tokenizer.decode(tokens[0], skip_special_tokens=True))
  # print(tokenizer.decode(hf_tokens[0], skip_special_tokens=True))
