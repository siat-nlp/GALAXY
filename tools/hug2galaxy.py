import torch
from collections import OrderedDict

BERT_VOCAB_SIZE = 30522


def get_match_value(name, state_dict_numpy):
    """
    Need be overridden towards different models, here for UnifiedTransformer Model
    """
    if name == 'embedder.token_embedding.weight':
        return state_dict_numpy['bert.embeddings.word_embeddings.weight']
    elif name == 'embedder.pos_embedding.weight':
        return state_dict_numpy['bert.embeddings.position_embeddings.weight']
    elif name == 'embedder.type_embedding.weight':
        return state_dict_numpy.get('bert.embeddings.token_type_embeddings.weight')
    elif name == 'embedder.turn_embedding.weight':
        return None
    elif name == 'embed_layer_norm.weight':
        return state_dict_numpy['bert.embeddings.LayerNorm.weight']
    elif name == 'embed_layer_norm.bias':
        return state_dict_numpy['bert.embeddings.LayerNorm.bias']
    elif name == 'pooler.0.weight':
        return state_dict_numpy.get('bert.pooler.dense.weight')
    elif name == 'pooler.0.bias':
        return state_dict_numpy.get('bert.pooler.dense.bias')
    elif name == 'mlm_transform.0.weight':
        return state_dict_numpy.get('cls.predictions.transform.dense.weight')
    elif name == 'mlm_transform.0.bias':
        return state_dict_numpy.get('cls.predictions.transform.dense.bias')
    elif name == 'mlm_transform.2.weight':
        return state_dict_numpy.get('cls.predictions.transform.LayerNorm.weight')
    elif name == 'mlm_transform.2.bias':
        return state_dict_numpy.get('cls.predictions.transform.LayerNorm.bias')
    elif name == 'mlm_bias':
        return state_dict_numpy.get('cls.predictions.bias')
    else:
        num = name.split('.')[1]
        assert num in [str(i) for i in range(12)]
        if name == f'layers.{num}.attn.linear_qkv.weight':
            q = state_dict_numpy[f'bert.encoder.layer.{num}.attention.self.query.weight']
            k = state_dict_numpy[f'bert.encoder.layer.{num}.attention.self.key.weight']
            v = state_dict_numpy[f'bert.encoder.layer.{num}.attention.self.value.weight']
            qkv_weight = torch.cat([q, k, v], dim=0)
            return qkv_weight
        elif name == f'layers.{num}.attn.linear_qkv.bias':
            q = state_dict_numpy[f'bert.encoder.layer.{num}.attention.self.query.bias']
            k = state_dict_numpy[f'bert.encoder.layer.{num}.attention.self.key.bias']
            v = state_dict_numpy[f'bert.encoder.layer.{num}.attention.self.value.bias']
            qkv_bias = torch.cat([q, k, v], dim=0)
            return qkv_bias
        elif name == f'layers.{num}.attn.linear_out.weight':
            return state_dict_numpy[f'bert.encoder.layer.{num}.attention.output.dense.weight']
        elif name == f'layers.{num}.attn.linear_out.bias':
            return state_dict_numpy[f'bert.encoder.layer.{num}.attention.output.dense.bias']
        elif name == f'layers.{num}.attn_norm.weight':
            return state_dict_numpy[f'bert.encoder.layer.{num}.attention.output.LayerNorm.weight']
        elif name == f'layers.{num}.attn_norm.bias':
            return state_dict_numpy[f'bert.encoder.layer.{num}.attention.output.LayerNorm.bias']
        elif name == f'layers.{num}.ff.linear_hidden.0.weight':
            return state_dict_numpy[f'bert.encoder.layer.{num}.intermediate.dense.weight']
        elif name == f'layers.{num}.ff.linear_hidden.0.bias':
            return state_dict_numpy[f'bert.encoder.layer.{num}.intermediate.dense.bias']
        elif name == f'layers.{num}.ff.linear_out.weight':
            return state_dict_numpy[f'bert.encoder.layer.{num}.output.dense.weight']
        elif name == f'layers.{num}.ff.linear_out.bias':
            return state_dict_numpy[f'bert.encoder.layer.{num}.output.dense.bias']
        elif name == f'layers.{num}.ff_norm.weight':
            return state_dict_numpy[f'bert.encoder.layer.{num}.output.LayerNorm.weight']
        elif name == f'layers.{num}.ff_norm.bias':
            return state_dict_numpy[f'bert.encoder.layer.{num}.output.LayerNorm.bias']
        else:
            raise ValueError(f'ERROR: Param "{name}" can not be loaded in Space Model!')


def convert(input_file, input_template, output_file):
    state_dict_output = OrderedDict()
    state_dict_input = torch.load(input_file, map_location=lambda storage, loc: storage)
    state_dict_template = torch.load(input_template, map_location=lambda storage, loc: storage)

    for name, value in state_dict_template.items():
        match_value = get_match_value(name, state_dict_input)
        if match_value is not None:
            assert match_value.ndim == value.ndim
            if match_value.shape != value.shape:
                assert value.size(0) == BERT_VOCAB_SIZE and match_value.size(0) > BERT_VOCAB_SIZE
                match_value = match_value[:BERT_VOCAB_SIZE]
            dtype = value.dtype
            device = value.device
            state_dict_output[name] = torch.tensor(match_value, dtype=dtype, device=device)
        else:
            print(f'WARNING: Param "{name}" can not be loaded in Space Model.')

    torch.save(state_dict_output, output_file)


if __name__ == '__main__':
    input_file = '../model/ToD-BERT-jnt/pytorch_model.bin'
    input_template = '../model/template.model'
    output_file = '../model/todbert.model'
    # output_file = '../model/unilm.model'
    # output_file = '../model/bert-base-uncased.model'
    # output_file = '../model/roberta-base.model'

    convert(input_file=input_file, input_template=input_template, output_file=output_file)
