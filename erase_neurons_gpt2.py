import torch
import pandas as pd
from pandas import DataFrame
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from functools import partial
from typing import Dict, List
from collections import defaultdict


def erase_neuron_embedding_hook(module, input, output, indexes):
    output[:, :, indexes] = 0


def erase_neuron_mlp_hook(module, input, output, indexes):
    output_attn = module.attn(module.ln_1(input[0]),
                              layer_past=None,
                              attention_mask=None,
                              head_mask=None)
    a = output_attn[0]  # output_attn: a, present, (attentions)

    x = input[0] + a
    m = module.mlp(module.ln_2(x))
    m[:, :, indexes] = 0
    x = x + m
    output = [x] + output_attn[1:]


def register_hooks(model, layer_neuron_list_dict):
    """
    :param model:
    :param layer_neuron_list_dict: dict in the form {num_layer: [neuron_index_list]}
    :return:
    """
    for layer, neurons in layer_neuron_list_dict.items():
        # EMBEDDING NEURON
        if layer == -1:
            model.transformer.wte.register_forward_hook(
                partial(
                    erase_neuron_embedding_hook,
                    indexes=neurons
                )
            )
        # MLP NEURON
        else:
            model.transformer.h[layer].register_forward_hook(
                partial(
                    erase_neuron_mlp_hook,
                    indexes=neurons
                )
            )


def get_layer_neurons_topk_dict(df: DataFrame, top_k=5) -> Dict[int, List]:
    df = df.sort_values(by=['total_causal_effect_mean'], ascending=False)
    layer_neurons_dict = defaultdict(list)
    for ind, neuron_loc in df[:top_k]["neuron_"].iteritems():
        print(neuron_loc)
        layer_str, neuron_str = neuron_loc.split('-')
        layer_neurons_dict[int(layer_str) - 1].append(int(neuron_str))

    return layer_neurons_dict


def main(
        csv_path,
        model_type='distilgpt2',
):

    model = GPT2LMHeadModel.from_pretrained(
        model_type,
        output_attentions=False)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # CREATE EXAMPLE INPUT
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_sentence = [tokenizer.encode("I saw a great movie yesterday")]
    input_sentence = torch.LongTensor(input_sentence).to(device)
    # REGISTER HOOKS

    neuro_effects_df = pd.read_csv(csv_path, index_col=0)
    layer_neuron_dict = get_layer_neurons_topk_dict(neuro_effects_df)
    register_hooks(model, layer_neuron_dict)
    output_after_hook = model(input_sentence)

    x = 1


if __name__ == '__main__':
    csv_path = "results/newest_intervention/distilgpt2_neuron_effects.csv"
    main(csv_path=csv_path)

