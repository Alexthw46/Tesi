import numpy as np
import torch

try:
	from src.adapter_chils import CHiLSAdapter
	from src.zshot_utils import get_CLIP_inputs_from_dict, conf_pred, conf_pred_hat, conf_pred_supagg
except ImportError:
	from src.adapter_chils import CHiLSAdapter
	from src.zshot_utils import get_CLIP_inputs_from_dict, conf_pred, conf_pred_hat, conf_pred_supagg


# CHiLS uses the ImageNet template bank for CIFAR20 (USETEMPLATES['cifar20'] = 'imagenet').
TEMPLATES_CIFAR20 = [
	'a bad photo of a {}.',
	'a photo of many {}.',
	'a sculpture of a {}.',
	'a photo of the hard to see {}.',
	'a low resolution photo of the {}.',
	'a rendering of a {}.',
	'graffiti of a {}.',
	'a bad photo of the {}.',
	'a cropped photo of the {}.',
	'a tattoo of a {}.',
	'the embroidered {}.',
	'a photo of a hard to see {}.',
	'a bright photo of a {}.',
	'a photo of a clean {}.',
	'a photo of a dirty {}.',
	'a dark photo of the {}.',
	'a drawing of a {}.',
	'a photo of my {}.',
	'the plastic {}.',
	'a photo of the cool {}.',
	'a close-up photo of a {}.',
	'a black and white photo of the {}.',
	'a painting of the {}.',
	'a painting of a {}.',
	'a pixelated photo of the {}.',
	'a sculpture of the {}.',
	'a bright photo of the {}.',
	'a cropped photo of a {}.',
	'a plastic {}.',
	'a photo of the dirty {}.',
	'a jpeg corrupted photo of a {}.',
	'a blurry photo of the {}.',
	'a photo of the {}.',
	'a good photo of the {}.',
	'a rendering of the {}.',
	'a {} in a video game.',
	'a photo of one {}.',
	'a doodle of a {}.',
	'a close-up photo of the {}.',
	'a photo of a {}.',
	'the origami {}.',
	'the {} in a video game.',
	'a sketch of a {}.',
	'a doodle of the {}.',
	'a origami {}.',
	'a low resolution photo of a {}.',
	'the toy {}.',
	'a rendition of the {}.',
	'a photo of the clean {}.',
	'a photo of a large {}.',
	'a rendition of a {}.',
	'a photo of a nice {}.',
	'a photo of a weird {}.',
	'a blurry photo of a {}.',
	'a cartoon {}.',
	'art of a {}.',
	'a sketch of the {}.',
	'a embroidered {}.',
	'a pixelated photo of a {}.',
	'itap of the {}.',
	'a jpeg corrupted photo of the {}.',
	'a good photo of a {}.',
	'a plushie {}.',
	'a photo of the nice {}.',
	'a photo of the small {}.',
	'a photo of the weird {}.',
	'the cartoon {}.',
	'art of the {}.',
	'a drawing of the {}.',
	'a photo of the large {}.',
	'a black and white photo of a {}.',
	'the plushie {}.',
	'a dark photo of a {}.',
	'itap of a {}.',
	'graffiti of the {}.',
	'a toy {}.',
	'itap of my {}.',
	'a photo of a cool {}.',
	'a photo of a small {}.',
	'a tattoo of the {}.',
]


def run_chils(clip_with_prompt, features, labels, sub2super, breeds_classes, reweighter='normal', experiment='true', templates=None, best_poss=False):
	"""Run a CHiLS-style evaluation using a CLIPWithL2P instance.

	Parameters
	- clip_with_prompt: instance with methods `encode_image_with_prompt` and CLIP backbone
	- features: torch.Tensor or numpy array. If 4D => image tensors; if 2D => precomputed image features
	- labels: 1D array of ground-truth superclass indices
	- sub2super: dict mapping superclass -> [list of subclass names]
	- breeds_classes: ordered list of superclass names (must match labels indices)
	- reweighter: 'normal'|'hat'|'supagg'
	- experiment: 'true' or 'true_lin' currently supported

	Returns (out_d, preds_d)
	"""
	raw_classes, reset_raw_to_super_mapping = get_CLIP_inputs_from_dict(sub2super, breeds_classes)

	if reweighter == 'normal':
		conf_func = conf_pred
	elif reweighter == 'hat':
		conf_func = conf_pred_hat
	elif reweighter == 'supagg':
		conf_func = conf_pred_supagg
	else:
		raise NotImplementedError(f"Unknown reweighter: {reweighter}")

	adapter = CHiLSAdapter(clip_with_prompt)
	if templates is None:
		templates = TEMPLATES_CIFAR20

	# superclass prototypes and logits
	adapter.zeroshot_classifier(breeds_classes, templates)
	super_out = adapter.emb_forward(features)['logits']
	super_preds = torch.argmax(super_out, dim=1).detach().cpu().numpy()

	if experiment in ['true', 'gpt']:
		adapter.zeroshot_classifier(raw_classes, templates)
		raw_out = adapter.emb_forward(features)['logits']

		conf1_preds = torch.argmax(conf_func(raw_out, super_out, reset_raw_to_super_mapping), dim=1).detach().cpu().numpy()
		raw_preds = torch.argmax(raw_out, dim=1).detach().cpu().numpy()
		raw_preds = np.array([reset_raw_to_super_mapping[x] for x in raw_preds])
		if reweighter != 'supagg':
			conf_preds = np.array([reset_raw_to_super_mapping[x] for x in conf1_preds])
		else:
			conf_preds = conf1_preds

		sup_01 = super_preds == labels
		conf_01 = conf_preds == labels

		out_d = {
			'Superclass': (super_preds == labels).sum() / len(labels),
			'CHiLSNoRW': (raw_preds == labels).sum() / len(labels),
			'CHiLS': (conf_preds == labels).sum() / len(labels)
		}
		if best_poss:
			out_d['Best'] = np.logical_or((super_preds == labels), (raw_preds == labels)).sum() / len(labels)

		preds_d = {}
		preds_d['both_wrong_idx'] = np.where((sup_01 == 0) & (conf_01 == 0))[0]
		preds_d['both_right_idx'] = np.where((sup_01 == 1) & (conf_01 == 1))[0]
		preds_d['chils_wrong_idx'] = np.where((sup_01 == 1) & (conf_01 == 0))[0]
		preds_d['sup_wrong_idx'] = np.where((sup_01 == 0) & (conf_01 == 1))[0]
		idx2sup = {i: x for i, x in enumerate(breeds_classes)}
		preds_d['labels'] = [idx2sup[x] for x in labels]
		preds_d['super_preds'] = [idx2sup[x] for x in super_preds]
		preds_d['conf_preds'] = [idx2sup[x] for x in conf_preds]
		idx2sub = {i: x for i, x in enumerate(raw_classes)}
		preds_d['conf_subpreds'] = [idx2sub[x] for x in conf1_preds]
		return out_d, preds_d
	else:
		assert experiment in ['true_lin', 'gpt_lin']
		# linear ensembling of subclass prototypes into superclass prototypes
		adapter.zeroshot_classifier_ens(sub2super.values(), templates)
		raw_out = adapter.emb_forward(features)['logits']
		raw_preds = torch.argmax(raw_out, dim=1).detach().cpu().numpy()
		out_d = {
			'Superclass': (super_preds == labels).sum() / len(labels),
			'CHiLSNoRW': (raw_preds == labels).sum() / len(labels),
		}
		return out_d, {}


