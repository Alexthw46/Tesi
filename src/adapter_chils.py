import torch
import clip
import numpy as np

class CHiLSAdapter:
    """Adapter that provides a minimal interface similar to the CLIP wrapper used in CHILS.

    It wraps a CLIPWithL2P-like model (prompt-tuned model) and exposes methods to build
    text prototypes (zeroshot_classifier variants) and to compute logits from image inputs
    via `emb_forward`.
    """
    def __init__(self, clip_with_prompt):
        self.model = clip_with_prompt
        # place-holder for current text prototypes (torch tensor, normalized)
        self.text_features = None
        self.temp_map = None

    def _build_text_features(self, classnames, templates):
        # templates: list of templates
        templates = [templates] if isinstance(templates, str) else list(templates)
        texts = [t.format(c) for c in classnames for t in templates]
        tokens = clip.tokenize(texts).to(self.model.device)
        with torch.no_grad():
            tfeats = self.model.bb.encode_text(tokens)
        tfeats = tfeats / tfeats.norm(dim=-1, keepdim=True)
        if len(templates) == 1:
            return tfeats
        n_classes = len(classnames)
        tfeats = tfeats.view(n_classes, len(templates), -1).mean(dim=1)
        tfeats = tfeats / tfeats.norm(dim=-1, keepdim=True)
        return tfeats

    def zeroshot_classifier(self, classnames, templates):
        self.text_features = self._build_text_features(classnames, templates)
        return self.text_features

    def zeroshot_classifier_set_templates(self, classnames, templates):
        # same behavior for our purposes
        return self.zeroshot_classifier(classnames, templates)

    def zeroshot_classifier_ens(self, list_of_classname_lists, templates):
        # list_of_classname_lists: iterable where each element is a list of subclass names
        prototypes = []
        for subs in list_of_classname_lists:
            if isinstance(subs, (list, tuple)):
                # build features for each subclass and average
                sub_feats = self._build_text_features(subs, templates)
                proto = sub_feats.mean(dim=0)
            else:
                proto = self._build_text_features([subs], templates).squeeze(0)
            proto = proto / proto.norm()
            prototypes.append(proto)
        self.text_features = torch.stack(prototypes, dim=0)
        return self.text_features

    def emb_forward(self, images_or_feats):
        """Compute logits given either raw images (4D tensor) or precomputed image features (2D tensor).

        Returns a dict { 'logits': logits_tensor }
        """
        with torch.no_grad():
            if isinstance(images_or_feats, np.ndarray):
                x = torch.tensor(images_or_feats)
            else:
                x = images_or_feats

            if x.dim() == 4:
                # image batch
                img_feats = self.model.encode_image_with_prompt(x.to(self.model.device))
                # ensure normalized
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            elif x.dim() == 2:
                # already image features
                img_feats = x.to(self.model.device)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            else:
                raise ValueError("emb_forward expects a 4D image tensor or 2D feature tensor")

            if self.text_features is None:
                raise RuntimeError("text_features not set. Call zeroshot_classifier first.")

            logits = 100.0 * (img_feats @ self.text_features.T)
            return { 'logits': logits }

