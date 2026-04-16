import torch
import torch.nn as nn
import clip
import json
import openai
import os
import time

openai.api_key = os.environ.get("OPENAI_API_KEY")

"""
CLIP-based zero-shot classification utilities.

This module provides a lightweight wrapper around OpenAI's CLIP model to
construct zero-shot text prototypes for image classification tasks. It also
includes utilities to query OpenAI completion models to generate descriptive
text prompts (used by the CuPL-like approach).

Key features:
- Build zero-shot classifier weights using prompt templates (averaging text
  embeddings across templates).
- CuPL-style generation of multiple textual descriptions per class via GPT
  completions (cached per class).
- Support for class-synonym ensembles and storing per-template embeddings.

Expected usage:
- Create a ZeroShotClip instance with a desired CLIP backbone (e.g., 'ViT-B/32').
- Call `init_text(dataset)` to initialize `self.text_features` for a dataset
  (currently mapped to ImageNet 1k classes or other supported keys).
- Use the forward/emb_forward methods to compute logits against initialized
  text prototypes.
"""


def query_gpt_prompt(prompt):
    """
    Query the OpenAI Completion API to produce a short descriptive sentence.

    This helper wraps the OpenAI Completion endpoint to return a single,
    trimmed sentence ending with a period. The method is used by
    `ZeroShotClip.zeroshot_classifier_cupl` to produce multiple natural-language
    descriptions of each class (CuPL-style).

    Args:
        prompt (str): Natural-language prompt to send to the completion model.

    Returns:
        str: A single descriptive sentence (trimmed, ending with '.').
    """
    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.99,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["."]
    )
    sub_str = completion.choices[0].text
    return sub_str.strip("\n") + '.'


class ZeroShotClip(nn.Module):
    """
        Wrapper around a CLIP model for zero-shot classification.

        This class loads a CLIP backbone and provides utilities to construct
        text-based prototype vectors for classes using prompt templates,
        GPT-generated descriptions, and synonym/ensemble strategies.

        Attributes:
            bb: The loaded CLIP model (both image and text encoders).
            preprocess: The CLIP preprocessing function (PIL transforms).
            templates (list[str]): A large list of prompt templates for ensembling.
            small_templates (list[str]): A shorter list for faster experiments.
            text_features (torch.Tensor): The text prototype matrix produced by
                `init_text` or other classifier-building methods. Shape is
                [embedding_dim, num_classes] after zeroshot generation.
        """

    def __init__(self, backbone="ViT-B/32", output_size=10, ):
        """
                Initialize the ZeroShotClip wrapper.

                Args:
                    backbone (str): CLIP backbone specifier passed to `clip.load`.
                        Examples: 'ViT-B/32', 'RN50', 'ViT-L/14@336px'.
                    output_size (int): Retained for API compatibility; not directly
                        used in zero-shot logic (prototypes drive logits dimension).
                """
        super(ZeroShotClip, self).__init__()
        # Load CLIP backbone and its preprocessing function. `clip.load`
        # will download model weights the first time if not present locally.
        self.bb, self.preprocess = clip.load(backbone)

        # A large ensemble of prompts used to build robust class prototypes
        # by averaging encoded template embeddings for each class string.

        self.templates = [
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
        # Smaller, faster-to-encode subset of templates suitable for quick tests.
        self.small_templates = [
            'a bad photo of a {}.',
            'a photo of a {}.',
            'a good photo of a {}.',
            'a sketch of the {}.',
            'a cropped photo of the {}.'
        ]


def zeroshot_classifier(self, classnames, templates):
    """
            Build zero-shot classifier prototypes by averaging encoded prompts.

            For each class name, this method formats all provided templates, tokenizes
            them with CLIP's tokenizer, encodes them with CLIP's text encoder, L2
            normalizes each template embedding, averages them, and re-normalizes
            the resulting class embedding. The returned tensor stacks class
            embeddings along the second axis and is moved to CUDA.

            Args:
                classnames (list[str]): List of class name strings (one per class).
                templates (list[str]): List of templates containing '{}' placeholder.

            Returns:
                torch.Tensor: Text feature matrix of shape [embedding_dim, num_classes]
                              on CUDA, suitable for dot-product similarity with
                              normalized image embeddings.
            """

    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = self.bb.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def zeroshot_classifier_cupl(self, classnames, typ):
    """
            Build zero-shot prototypes using GPT-generated class descriptions (CuPL-style).

            This method uses `query_gpt_prompt` to generate one or more natural language
            descriptions for each class. Descriptions for each class are cached in
            `self.class_texts` to avoid repeated API calls. The method collects up to
            50 descriptions per class (generating additional ones on subsequent calls),
            tokenizes them, encodes with CLIP's text encoder, averages, normalizes, and
            stacks the resulting embeddings.

            NOTE: This method makes external API calls to OpenAI and requires
            OPENAI_API_KEY to be set in the environment. Use cautiously to avoid
            excessive API usage/costs.

            Args:
                classnames (list[str]): List of class name strings.
                typ (str): A higher-level category string inserted into the prompt
                    (e.g., "animal", "vehicle") to make descriptions context-aware.

            Returns:
                torch.Tensor: Text feature matrix [embedding_dim, num_classes] on CUDA.
            """
    if getattr(self, 'class_texts', None) is None:
        self.class_texts = {}
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            if self.class_texts.get(classname) is None:
                self.class_texts[classname] = [
                    query_gpt_prompt(f'Describe what a(n) {classname}, a type of {typ}, looks like:')]
            else:
                while len(self.class_texts[classname]) < 50:
                    self.class_texts[classname].append(
                        query_gpt_prompt(f'Describe what a(n) {classname}, a type of {typ}, looks like:'))
            # texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(self.class_texts[classname]).cuda()  # tokenize
            class_embeddings = self.bb.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def zeroshot_classifier_ens(self, classnames, templates):
    """
            Build zero-shot prototypes when each class has multiple synonym strings.

            Each element in `classnames` can be a list of alias strings for that class.
            This method formats each template with each alias, tokenizes and encodes
            the expanded text list, and averages across all generated embeddings for
            the class.

            Args:
                classnames (list[list[str]]): Per-class list of synonyms/aliases.
                templates (list[str]): List of templates containing '{}' placeholder.

            Returns:
                torch.Tensor: Text feature matrix [embedding_dim, num_classes] on CUDA.
            """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [[template.format(x) for x in classname] for template in templates]  # format with class
            texts = [item for sublist in texts for item in sublist]
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = self.bb.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def zeroshot_classifier_set_templates(self, classnames, templates):
    """
            Return text embeddings for every (class, template) pair and create a mapping.

            Instead of averaging across templates per class, this method preserves each
            template's embedding separately and concatenates them. It also builds
            `self.temp_map` to map each template-indexed embedding back to the
            corresponding class index, which can be useful for template-level scoring
            or custom aggregation strategies.

            Args:
                classnames (list[str]): List of class name strings.
                templates (list[str]): List of templates containing '{}' placeholder.

            Returns:
                torch.Tensor: Matrix of embeddings transposed to shape
                              [embedding_dim, num_classes * num_templates] on CUDA.
            """

    self.temp_map = {}
    with torch.no_grad():
        zeroshot_weights = []
        for idx, classname in enumerate(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = self.bb.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embedding = class_embeddings.mean(dim=0)
            # class_embedding /= class_embedding.norm()
            for sub_idx, template in enumerate(templates):
                self.temp_map[idx * len(templates) + sub_idx] = idx
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).cuda().T
    return zeroshot_weights


def zeroshot_classifier_prompt(self, classnames, prompt_vectors):
    """
    Build zero-shot prototypes by adding continuous prompt vectors to base
    class embeddings. This is a light-weight alternative to inserting
    learned tokens into the CLIP tokenizer: we compute a base class
    embedding using the provided templates and then add a learned
    continuous vector (in the embedding space) per class.

    Args:
        classnames (list[str]): List of class name strings.
        prompt_vectors (torch.Tensor or numpy): If a tensor, expected shape
            [num_classes, embed_dim]. If numpy, it will be converted.

    Returns:
        torch.Tensor: Text feature matrix [embed_dim, num_classes] on CUDA.
    """
    # compute base embeddings using the full template set (keeps behavior similar to zeroshot)
    with torch.no_grad():
        base = self.zeroshot_classifier(classnames, self.templates)  # [embed_dim, num_classes]
        base = base.t()  # [num_classes, embed_dim]
        if not isinstance(prompt_vectors, torch.Tensor):
            prompt_vectors = torch.tensor(prompt_vectors, device=base.device, dtype=base.dtype)
        else:
            prompt_vectors = prompt_vectors.to(base.device)

        assert prompt_vectors.shape[0] == base.shape[0], "prompt_vectors must have same first-dim as number of classes"
        # add and normalize per-class
        prot = base + prompt_vectors
        prot = prot / prot.norm(dim=-1, keepdim=True)
        prot = prot.t().contiguous().cuda()
    return prot


def prompt_tune(self, image_features, labels, classnames, base_templates=None,
                epochs=50, lr=1e-2, batch_size=128, device='cuda', verbose=False):
    """
    Train continuous prompt vectors (one vector per class) in the text-embedding
    space to adapt class prototypes to provided image features.

    This implementation is intentionally simple: the CLIP encoders remain
    frozen and we learn an additive vector per class in the same space as the
    CLIP text embeddings. Optimization minimizes cross-entropy between the
    similarity logits and the provided labels.

    Args:
        image_features (np.array or torch.Tensor): Precomputed image embeddings
            (N x D) compatible with CLIP image encoder outputs.
        labels (np.array or torch.Tensor): Integer labels (N,) matching classnames order.
        classnames (list[str]): Ordered list of class names.
        base_templates (list[str], optional): Templates used to compute base
            class embeddings. Defaults to self.small_templates if None.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Mini-batch size.
        device (str or torch.device): Device to run training on.
        verbose (bool): If True prints loss per epoch.

    Returns:
        torch.Tensor: Learned prompt vectors (num_classes x embed_dim) on CPU.
    """
    # Choose device: use CUDA only if requested and available, otherwise CPU
    if device == 'cuda' and torch.cuda.is_available():
        torch_device = torch.device('cuda')
    else:
        torch_device = torch.device('cpu')
    if not isinstance(image_features, torch.Tensor):
        image_features = torch.tensor(image_features, dtype=torch.float32, device=torch_device)
    else:
        image_features = image_features.to(torch_device).float()
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long, device=torch_device)
    else:
        labels = labels.to(torch_device)

    templates = base_templates if base_templates is not None else self.small_templates

    # compute base embeddings (on device)
    with torch.no_grad():
        base = self.zeroshot_classifier(classnames, templates)  # [embed_dim, num_classes]
    base = base.t().to(torch_device)  # [num_classes, embed_dim]

    num_classes, embed_dim = base.shape[0], base.shape[1]

    # initialize prompt vectors to small random values
    prompts = torch.nn.Parameter(torch.zeros((num_classes, embed_dim), device=torch_device))
    optimizer = torch.optim.Adam([prompts], lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    N = image_features.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N, device=torch_device)
        epoch_loss = 0.0
        iters = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = image_features[idx]
            yb = labels[idx]

            xb_n = xb / xb.norm(dim=-1, keepdim=True)
            prototypes = base + prompts  # [num_classes, embed_dim]
            prototypes_n = prototypes / prototypes.norm(dim=-1, keepdim=True)
            logits = 100.0 * (xb_n @ prototypes_n.t())
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iters += 1

        if verbose:
            print(f"Prompt-tune epoch {ep+1}/{epochs} loss={epoch_loss/iters:.4f}")

    # finalize: set self.text_features to learned prototypes (embedding_dim x num_classes)
    with torch.no_grad():
        final_prot = (base + prompts).t()
        final_prot = final_prot / final_prot.norm(dim=0, keepdim=True)
        self.text_features = final_prot.cuda() if torch_device.type == 'cuda' else final_prot
        # keep CPU copy of learned prompts for later use
        self.prompt_vectors = prompts.detach().cpu()

    return self.prompt_vectors


def init_text(self, dataset):
    """
            Initialize `self.text_features` for a supported dataset identifier.

            Currently, this implementation maps a variety of dataset strings to the
            ImageNet-1k class names (a long list embedded below). If `dataset` is
            recognized, the method calls `zeroshot_classifier` using the module's
            `self.templates` to produce `self.text_features`. Otherwise, a
            NotImplementedError is raised.

            Args:
                dataset (str): Dataset key name used to select a class vocabulary.

            Raises:
                NotImplementedError: If the dataset key is not implemented.
            """
    if dataset in ['imagenet', 'imagenet-sketch', 'imagenet-c1', 'imagenet-c2', 'imagenet-c3', 'imagenet-c4',
                   'imagenet-c5', 'imagenetv2', 'officehome-product', 'officehome-realworld', 'cifar100',
                   'fruits360', 'fashion1M', 'fashion1m', 'food-101', 'lsun-scene', 'office31-amazon',
                   'office31-dslr', 'office31-webcam', 'fashion-mnist', 'objectnet', 'officehome-art',
                   'officehome-clipart', 'resisc45', 'eurosat']:
        classes = [
            "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray",
            "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
            "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)",
            "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt",
            "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog",
            "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle",
            "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama",
            "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon",
            "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake",
            "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake",
            "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
            "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
            "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider",
            "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede",
            "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge",
            "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater",
            "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan",
            "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone",
            "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton",
            "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab",
            "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork",
            "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin",
            "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank",
            "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale",
            "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu",
            "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound",
            "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound",
            "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound",
            "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki",
            "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
            "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier",
            "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier",
            "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier",
            "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
            "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso",
            "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever",
            "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter",
            "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel",
            "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz",
            "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor",
            "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog",
            "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog",
            "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff",
            "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute",
            "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
            "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
            "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle",
            "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf",
            "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox",
            "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard",
            "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear",
            "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle",
            "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
            "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing",
            "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly",
            "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin",
            "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel",
            "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog",
            "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep",
            "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink",
            "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth",
            "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon",
            "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
            "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
            "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda",
            "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish",
            "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar",
            "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock",
            "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon",
            "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop",
            "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon",
            "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker",
            "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib",
            "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh",
            "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie",
            "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle",
            "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon",
            "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
            "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
            "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
            "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church",
            "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug",
            "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store",
            "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle",
            "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch",
            "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock",
            "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome",
            "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan",
            "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine",
            "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole",
            "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
            "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator",
            "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano",
            "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track",
            "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica",
            "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt",
            "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron",
            "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono",
            "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap",
            "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick",
            "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass",
            "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba",
            "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
            "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
            "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped",
            "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent",
            "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace",
            "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter",
            "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging",
            "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel",
            "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone",
            "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum",
            "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow",
            "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium",
            "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
            "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer",
            "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car",
            "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle",
            "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver",
            "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker",
            "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale",
            "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt",
            "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket",
            "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag",
            "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser",
            "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar",
            "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car",
            "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
            "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine",
            "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt",
            "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player",
            "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble",
            "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch",
            "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat",
            "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile",
            "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase",
            "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin",
            "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
            "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig",
            "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon",
            "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword",
            "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme",
            "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog",
            "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash",
            "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple",
            "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)",
            "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie",
            "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
            "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
            "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip",
            "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus",
            "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
        self.text_features = self.zeroshot_classifier(classes, self.templates)
    else:
        raise NotImplementedError(f"{dataset} not implemented for Clip Zero Shot")


def forward(self, x):
    """
           Forward pass: encode images and compute similarity logits vs text prototypes.

           The method encodes the input images using CLIP's image encoder (no gradient
           updates), L2 normalizes the image embeddings, computes dot-product
           similarity with pre-computed `self.text_features` (scaled by 100),
           and returns both logits and raw image features.

           Args:
               x (torch.Tensor): A batch of input images already preprocessed for
                                 CLIP's image encoder (e.g., using `self.preprocess`).

           Returns:
               dict: {
                   'logits': similarity scores (torch.Tensor) between image and text
                             prototypes,
                   'features': raw image embedding tensor from CLIP's image encoder
               }
           """
    with torch.no_grad():
        # x = self.preprocess(x)
        image_features = self.bb.encode_image(x)
    image_features_n = image_features / image_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features_n @ self.text_features)
    output = {
        'logits': similarity,
        'features': image_features
    }
    return output


def emb_forward(self, image_features):
    """
            Compute logits directly from precomputed image embeddings.

            Normalizes the given image embeddings and computes similarity to
            `self.text_features`. Useful when image embeddings were computed
            elsewhere or cached for efficiency.

            Args:
                image_features (torch.Tensor): One or more image embeddings from
                                               CLIP's image encoder.

            Returns:
                dict: {
                    'logits': similarity scores (torch.Tensor),
                    'features': the original image_features tensor (passthrough)
                }
            """
    image_features_n = image_features / image_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features_n @ self.text_features)
    output = {
        'logits': similarity,
        'features': image_features
    }
    return output


def ClipRN50(num_classes=10):
    """
        Factory helper to construct ZeroShotClip with ResNet-50 CLIP backbone.

        Args:
            num_classes (int): Kept for API compatibility; not used directly.

        Returns:
            ZeroShotClip: Instance with 'RN50' backbone.
        """
    return ZeroShotClip('RN50', output_size=num_classes)


def ClipRN101(num_classes=10):
    return ZeroShotClip('RN101', output_size=num_classes)


def ClipRN50x4(num_classes=10):
    return ZeroShotClip( 'RN50x4', output_size=num_classes)


def ClipViTB16(num_classes=10):
    return ZeroShotClip('ViT-B/16', output_size=num_classes)


def ClipViTB32(num_classes=10):
    """
        Factory helper to construct ZeroShotClip with ViT-L/14@336px CLIP backbone.
        """
    return ZeroShotClip('ViT-B/32', output_size=num_classes)


def ClipViTL14(num_classes=10):
    return ZeroShotClip('ViT-L/14@336px', output_size=num_classes)
