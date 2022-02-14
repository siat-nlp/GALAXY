import os
import torch
from collections import OrderedDict


def get_match_value(name, state_dict_numpy):
    """
    Need be overridden towards different models, here for UnifiedTransformer Model
    """
    if name == 'bert.embeddings.word_embeddings.weight':
        return state_dict_numpy['embedder.token_embedding.weight']
    elif name == 'bert.embeddings.position_embeddings.weight':
        return state_dict_numpy['embedder.pos_embedding.weight']
    elif name == 'bert.embeddings.token_type_embeddings.weight':
        return state_dict_numpy['embedder.type_embedding.weight']
    elif name == 'bert.embeddings.LayerNorm.weight':
        return state_dict_numpy['embed_layer_norm.weight']
    elif name == 'bert.embeddings.LayerNorm.bias':
        return state_dict_numpy['embed_layer_norm.bias']
    elif name == 'bert.pooler.dense.weight':
        return state_dict_numpy['pooler.0.weight']
    elif name == 'bert.pooler.dense.bias':
        return state_dict_numpy['pooler.0.bias']
    elif name == 'cls.predictions.transform.dense.weight':
        return state_dict_numpy['mlm_transform.0.weight']
    elif name == 'cls.predictions.transform.dense.bias':
        return state_dict_numpy['mlm_transform.0.bias']
    elif name == 'cls.predictions.transform.LayerNorm.weight':
        return state_dict_numpy['mlm_transform.2.weight']
    elif name == 'cls.predictions.transform.LayerNorm.bias':
        return state_dict_numpy['mlm_transform.2.bias']
    elif name == 'cls.predictions.bias':
        return state_dict_numpy['mlm_bias']
    elif name == 'cls.predictions.decoder.weight':
        return state_dict_numpy['embedder.token_embedding.weight']
    elif name == 'cls.predictions.decoder.bias':
        return state_dict_numpy['mlm_bias']
    else:
        num = name.split('.')[3]
        assert num in [str(i) for i in range(12)]
        if name == f'bert.encoder.layer.{num}.attention.self.query.weight':
            qkv_weight = state_dict_numpy[f'layers.{num}.attn.linear_qkv.weight']
            return qkv_weight[:768]
        elif name == f'bert.encoder.layer.{num}.attention.self.key.weight':
            qkv_weight = state_dict_numpy[f'layers.{num}.attn.linear_qkv.weight']
            return qkv_weight[768: 1536]
        elif name == f'bert.encoder.layer.{num}.attention.self.value.weight':
            qkv_weight = state_dict_numpy[f'layers.{num}.attn.linear_qkv.weight']
            return qkv_weight[1536:]
        elif name == f'bert.encoder.layer.{num}.attention.self.query.bias':
            qkv_bias = state_dict_numpy[f'layers.{num}.attn.linear_qkv.bias']
            return qkv_bias[:768]
        elif name == f'bert.encoder.layer.{num}.attention.self.key.bias':
            qkv_bias = state_dict_numpy[f'layers.{num}.attn.linear_qkv.bias']
            return qkv_bias[768: 1536]
        elif name == f'bert.encoder.layer.{num}.attention.self.value.bias':
            qkv_bias = state_dict_numpy[f'layers.{num}.attn.linear_qkv.bias']
            return qkv_bias[1536:]
        elif name == f'bert.encoder.layer.{num}.attention.output.dense.weight':
            return state_dict_numpy[f'layers.{num}.attn.linear_out.weight']
        elif name == f'bert.encoder.layer.{num}.attention.output.dense.bias':
            return state_dict_numpy[f'layers.{num}.attn.linear_out.bias']
        elif name == f'bert.encoder.layer.{num}.attention.output.LayerNorm.weight':
            return state_dict_numpy[f'layers.{num}.attn_norm.weight']
        elif name == f'bert.encoder.layer.{num}.attention.output.LayerNorm.bias':
            return state_dict_numpy[f'layers.{num}.attn_norm.bias']
        elif name == f'bert.encoder.layer.{num}.intermediate.dense.weight':
            return state_dict_numpy[f'layers.{num}.ff.linear_hidden.0.weight']
        elif name == f'bert.encoder.layer.{num}.intermediate.dense.bias':
            return state_dict_numpy[f'layers.{num}.ff.linear_hidden.0.bias']
        elif name == f'bert.encoder.layer.{num}.output.dense.weight':
            return state_dict_numpy[f'layers.{num}.ff.linear_out.weight']
        elif name == f'bert.encoder.layer.{num}.output.dense.bias':
            return state_dict_numpy[f'layers.{num}.ff.linear_out.bias']
        elif name == f'bert.encoder.layer.{num}.output.LayerNorm.weight':
            return state_dict_numpy[f'layers.{num}.ff_norm.weight']
        elif name == f'bert.encoder.layer.{num}.output.LayerNorm.bias':
            return state_dict_numpy[f'layers.{num}.ff_norm.bias']
        else:
            raise ValueError('No matched name in state_dict_numpy!')


def numpy2pytorch(input_unilm, input_pt, output_pt, restore=True):
    state_dict_pytorch = OrderedDict()
    state_dict_init_unilm = torch.load(input_unilm, map_location=lambda storage, loc: storage)
    state_dict_init_pytorch = torch.load(input_pt, map_location=lambda storage, loc: storage)
    if 'module.' in list(state_dict_init_pytorch.keys())[0]:
        new_model_state_dict = OrderedDict()
        for k, v in state_dict_init_pytorch.items():
            assert k[:7] == 'module.'
            new_model_state_dict[k[7:]] = v
        state_dict_init_pytorch = new_model_state_dict

    for name, value in state_dict_init_unilm.items():
        match_value = get_match_value(name, state_dict_init_pytorch)
        if match_value is not None:
            assert match_value.shape == value.shape
            assert match_value.dtype == value.dtype
            state_dict_pytorch[name] = match_value
        else:
            print(f'{match_value} is not existed!')
            if restore:
                state_dict_pytorch[name] = value
            else:
                continue

    torch.save(state_dict_pytorch, output_pt)


if __name__ == '__main__':
    restore = True
    input_unilm = '../model/bert-base-uncased/pytorch_model.bin'
    for num in range(5, 11):
        input_pytorch = os.path.join('/data_hdd/myself/Space-XL/outputs/pre_train/MultiWOZ/convbert-super[false]-156-0-drop0.2-mlm0.1-mmd0.1-tem0.07-Pfalse-Gfalse-ppu5-ppp5-qbowfalse-rbowfalse-epoch60-lr1e-5-system-dstc2,dstc3,incar,MultiWOZ2.2,multiwoz_synthesis,SGD,taskmaster1,taskmaster2,taskmaster3,woz-seed11',
                                     f'state_epoch_{num}.model')

        output_dir = f'/home/myself/dialoglue/trippy/model/xl-super-multiwoz-{num}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_pytorch = os.path.join(output_dir, 'pytorch_model.bin')

        numpy2pytorch(input_unilm=input_unilm, input_pt=input_pytorch, output_pt=output_pytorch, restore=restore)
        print(f'Converted num {num}')
