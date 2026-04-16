import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Union, Sequence

from models.L2P import Prompt

class CLIPWithL2P(nn.Module):
    def __init__(self, device, classnames, template: Union[str, Sequence[str]], clip_model="ViT-B/16", prompt_lenght=5, prompt_pool_size=20):
        super().__init__()

        # --- CLIP backbone ---
        self.device = device
        self.bb, _ = clip.load(clip_model, device=device)

        # freeze CLIP
        for p in self.bb.parameters():
            p.requires_grad = False

        # --- Prompt module ---
        embed_dim = self.bb.visual.transformer.width
        self.prompt = Prompt(
            length=prompt_lenght,
            embed_dim=embed_dim,
            prompt_pool=True,
            pool_size=prompt_pool_size,
            top_k=5,
            prompt_key=True
        ).to(self.device)

        # --- Text features (zero-shot prototypes) ---
        self.classnames = classnames
        self.template = template
        self.text_features = self._build_text_features()

    def _build_text_features(self):
        templates = [self.template] if isinstance(self.template, str) else list(self.template)
        texts = [t.format(c) for c in self.classnames for t in templates]
        tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            text_features = self.bb.encode_text(tokens)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if len(templates) == 1:
            return text_features

        # Reduce multiple template features into one prototype per class.
        n_classes = len(self.classnames)
        text_features = text_features.view(n_classes, len(templates), -1).mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image_with_prompt(self, image):
        visual = self.bb.visual
        model_dtype = visual.conv1.weight.dtype

        # Match CLIP visual dtype/device (CUDA CLIP commonly runs in fp16).
        image = image.to(device=self.device, dtype=model_dtype)

        # --- patch embedding ---
        x = visual.conv1(image)                       # [B, C, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1)     # [B, C, N]
        x = x.permute(0, 2, 1)                        # [B, N, C]

        # --- CLS token ---
        cls_token = visual.class_embedding.to(x.dtype)
        cls_token = cls_token.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)

        patches = x
        cls = cls_token

        # --- prompt selection ---
        prompt_dtype = next(self.prompt.parameters()).dtype
        prompt_out = self.prompt(patches.to(prompt_dtype))
        prompts = prompt_out["prompted_embedding"][:, :prompt_out["total_prompt_len"], :].to(x.dtype)

        # --- assemble tokens ---
        x = torch.cat([cls, prompts, patches], dim=1)

        # --- positional embedding ---
        pos = visual.positional_embedding.to(device=x.device, dtype=x.dtype)

        if pos.shape[0] != x.shape[1]:
            pos = F.interpolate(
                pos.T.unsqueeze(0),
                size=x.shape[1],
                mode="linear",
                align_corners=False
            ).squeeze(0).T

        x = x + pos.unsqueeze(0)
        x = visual.ln_pre(x)

        # --- transformer ---
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        # --- CLS output ---
        x = x[:, 0, :]
        x = visual.ln_post(x)

        if visual.proj is not None:
            x = x @ visual.proj

        return x

    def forward(self, image):
        image_features = self.encode_image_with_prompt(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_features.to(device=image_features.device, dtype=image_features.dtype)
        logits = 100.0 * (image_features @ text_features.T)
        return logits

    def emb_forward(self, image):
        image_features = self.encode_image_with_prompt(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    @property
    def prompt_vectors(self):
        """Expose the underlying prompt parameters (if any) so external code can access learned prompts.

        Returns the raw prompt tensor when a prompt pool or prompt parameter is present, otherwise None.
        This allows evaluation code that checks `clip_mod.prompt_vectors` to work after loading a
        checkpoint that contains the prompt parameters.
        """
        # The Prompt module stores pool parameters as `self.prompt` when prompt_pool=True,
        # otherwise it stores a single prompt as `self.prompt` as well in the forward branch.
        return getattr(self.prompt, 'prompt', None)
