import re
import time

import openai
import torch
import torch.nn.functional as F


def query_gpt_prompt(prompt):
    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.15,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    sub_str = completion.choices[0].text
    # sub_list = [x.strip() for x in re.sub('[^a-zA-Z, ]+', '', sub_str).split(",")]
    return sub_str.strip("\n")


def get_CLIP_inputs_from_dict(cl_set_dict, ord_class):
    '''
    cl_set_dict: any dict that is of the form {class: [list of class words]}
    ord_class: ORDERED list of classnames that must match dataset
    '''
    # tot_subs = 0
    # subcs = []
    # for k,v in cl_set_dict.items():
    #     tot_subs += len(v)
    #     subcs += v
    # subclasses = [''] * tot_subs
    counter = 0
    sub_to_super = {}
    # subcs = set(subcs)
    subclasses = []
    for idx, cl in enumerate(ord_class):
        subs = cl_set_dict[cl]
        for sub in subs:
            subclasses.append(sub)
            assert subclasses[counter] == sub
            sub_to_super[counter] = idx
            counter += 1

    return subclasses, sub_to_super


def get_idir(data_dir, dist):
    if dist == 'imagenet':
        imagenet_dir = data_dir + "/imagenet/imagenetv1/train/"
    elif dist == 'imagenet-sketch':
        imagenet_dir = data_dir + "/imagenet/imagenet-sketch/sketch/"
    elif dist == 'imagenet-c1':
        imagenet_dir = data_dir + '/imagenet/imagenet-c/fog/1'
    elif dist == 'imagenet-c2':
        imagenet_dir = data_dir + '/imagenet/imagenet-c/contrast/2'
    elif dist == 'imagenet-c3':
        imagenet_dir = data_dir + '/imagenet/imagenet-c/snow/3'
    elif dist == 'imagenet-c4':
        imagenet_dir = data_dir + '/imagenet/imagenet-c/gaussian_blur/4'
    elif dist == 'imagenet-c5':
        imagenet_dir = data_dir + '/imagenet/imagenet-c/saturate/5'
    elif dist == 'imagenetv2':
        imagenet_dir = data_dir + '/imagenet/imagenetv2/imagenetv2-matched-frequency-format-val'
    return imagenet_dir


def query_gpt(word, n=10, temp=0.1, context=None):
    try:
        if context is not None:
            prompt = f"Generate a comma separated list of {n} types of the following {context}:\n\n>{word}:"
        else:
            prompt = f"Generate a comma separated list of {n} types of the following:\n\n>{word}:"

        completion = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=temp,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        sub_str = completion.choices[0].text
        sub_list = [x.strip() for x in re.sub('[^a-zA-Z, ]+', '', sub_str).split(",")]
        return sub_list
    except:
        time.sleep(10)
        return query_gpt(word, n, temp)


def get_cleaned_gpt_sets(sup_classes, n=10, temp=0.1, context=None):
    out_d = {}
    for cl in sup_classes:
        if out_d.get(cl) is None:
            out_d[cl] = query_gpt(cl, n=n, temp=temp, context=context)
    new_d = out_d.copy()
    for k, v in out_d.items():
        new_v = []
        if k not in v:
            new_v.append(k)
        for sub in v:
            if k.lower() not in sub.lower():
                new_v.append(sub + f" {k}")
            else:
                new_v.append(sub)
        new_d[k] = new_v
    return new_d

def conf_pred(raw_logits, super_logits, raw2super_map):
    inv_map = {}
    for k, v in raw2super_map.items():
        if inv_map.get(v) is None:
            inv_map[v] = [k]
        else:
            inv_map[v].append(k)

    for k, v in inv_map.items():
        inv_map[k] = sorted(v)
    raw_probs = F.softmax(raw_logits, dim=1)
    super_probs = F.softmax(super_logits, dim=1)
    output = torch.zeros_like(raw_probs)

    for out_cl, in_cls in inv_map.items():
        sub_probs = raw_probs[:, in_cls]

        if len(sub_probs.shape) == 1:
            sub_probs = sub_probs.reshape(-1, 1)

        raw_probs[:, in_cls] = raw_probs[:, in_cls] * super_probs[:, out_cl].reshape(-1, 1)
    return raw_probs


def conf_pred_hat(raw_logits, super_logits, raw2super_map):
    inv_map = {}
    for k, v in raw2super_map.items():
        if inv_map.get(v) is None:
            inv_map[v] = [k]
        else:
            inv_map[v].append(k)

    for k, v in inv_map.items():
        inv_map[k] = sorted(v)
    raw_probs = F.softmax(raw_logits, dim=1)
    super_probs = F.softmax(super_logits, dim=1)
    output = torch.zeros_like(raw_probs)

    for out_cl, in_cls in inv_map.items():
        sub_probs = raw_probs[:, in_cls]

        if len(sub_probs.shape) == 1:
            sub_probs = sub_probs.reshape(-1, 1)

        raw_sums = sub_probs.sum(axis=1).reshape(-1, 1)

        raw_probs[:, in_cls] = raw_probs[:, in_cls] * raw_sums
    return raw_probs


def conf_pred_supagg(raw_logits, super_logits, raw2super_map):
    inv_map = {}
    for k, v in raw2super_map.items():
        if inv_map.get(v) is None:
            inv_map[v] = [k]
        else:
            inv_map[v].append(k)

    for k, v in inv_map.items():
        inv_map[k] = sorted(v)
    raw_probs = F.softmax(raw_logits, dim=1)
    super_probs = F.softmax(super_logits, dim=1)
    output = torch.zeros_like(raw_probs)
    superagg_probs = torch.ones_like(super_probs)

    for out_cl, in_cls in inv_map.items():
        sub_probs = raw_probs[:, in_cls]

        if len(sub_probs.shape) == 1:
            sub_probs = sub_probs.reshape(-1, 1)

        raw_sums = sub_probs.sum(axis=1).reshape(-1, 1)
        superagg_probs[:, out_cl] = super_probs[:, out_cl] * raw_sums.flatten()
    return superagg_probs


