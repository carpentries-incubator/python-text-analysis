# Load model and retrieve attention weights
from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel

class AttentionViz:
  def __init__(self):
    model_version = 'bert-base-uncased'
    self.model = BertModel.from_pretrained(model_version, output_attentions=True)
    self.tokenizer = BertTokenizer.from_pretrained(model_version)

  #overload method to work with sentences as arguments
  def __init__(self, sentence_a, sentence_b):
    model_version = 'bert-base-uncased'
    self.model = BertModel.from_pretrained(model_version, output_attentions=True)
    self.tokenizer = BertTokenizer.from_pretrained(model_version)
    self.compare(sentence_a, sentence_b)

  #with compare we don't have to reinit when we change sentences
  def compare(self, sentence_a, sentence_b):
    inputs = self.tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    self.attention = self.model(input_ids, token_type_ids=token_type_ids)[-1]
    self.sentence_b_start = token_type_ids[0].tolist().index(1)
    input_id_list = input_ids[0].tolist() # Batch index 0
    self.tokens = self.tokenizer.convert_ids_to_tokens(input_id_list) 

  def hview(self):
    head_view(self.attention, self.tokens, self.sentence_b_start)

  def mview(self):
    model_view(self.attention, self.tokens, self.sentence_b_start)

  def nview(self):    
    #TODO- pull this initialization stuff out of the function if needed, not clear if it will be used as part of lesson though
    model_type = 'bert'
    model_version = 'bert-base-uncased'
    model = BertModel.from_pretrained(model_version, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
    show(model, model_type, tokenizer, sentence_a, sentence_b, layer=4, head=3)