import numpy as np

# Minimal constants ported from CHILS for CIFAR label-sets
CIFAR20_COARSE = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
    'household electrical devices', 'household furniture', 'insects', 'large carnivores',
    'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
    'medium mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees', 'vehicles'
]

CIFAR20_FINE = [
    'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear', 'pickup truck', 'pine tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
    'sunflower', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm'
]

# Mapping from the original CIFAR-100 fine class index to the CIFAR-20 coarse class index
CIFAR20_LABELS = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
            6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
            0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
            5, 18,  8,  8, 15, 13, 14, 17, 18, 10,
            16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
            10, 3,  2, 12, 12, 16, 12,  1,  9, 18,
            2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
            16, 18,  2,  4,  6, 18,  5,  5,  8, 18,
            18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

__all__ = [
    'CIFAR20_COARSE', 'CIFAR20_FINE', 'CIFAR20_LABELS'
]

