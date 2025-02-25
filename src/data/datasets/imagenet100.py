from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from PIL import Image

class Imagenet100(VisionDataset):
    """
    Imagenet100 is a subset of the original ImageNet-1k dataset containing 100 randomly selected classes.
    Introduced in `paper <https://arxiv.org/abs/1906.05849>` and `source code <https://github.com/HobbitLong/CMC>`. 
    Download manually from: https://www.kaggle.com/datasets/ambityga/imagenet100
    

    Train(train) Contains 1300 images for each class.
    Validation(val) contains 50 images for each class

    Args:
        root (str or ``pathlib.Path``): Root directory of the Imagenette dataset.
        split (string, optional): The dataset split. Supports ``"train"``, and ``"val"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version, e.g. ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
     
     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class name, class index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (WordNet ID, class index).
    """

    _WNID_TO_CLASS = {
        "n01968897": "chambered nautilus, pearly nautilus, nautilus",
        "n01770081": "harvestman, daddy longlegs, Phalangium opilio",
        "n01818515": "macaw", 
        "n02011460": "bittern", 
        "n01496331": "electric ray, crampfish, numbfish, torpedo", 
        "n01847000": "drake", 
        "n01687978": "agama", 
        "n01740131": "night snake, Hypsiglena torquata", 
        "n01537544": "indigo bunting, indigo finch, indigo bird, Passerina cyanea", 
        "n01491361": "tiger shark, Galeocerdo cuvieri", 
        "n02007558": "flamingo", 
        "n01735189": "garter snake, grass snake", 
        "n01630670": "common newt, Triturus vulgaris", 
        "n01440764": "tench, Tinca tinca", 
        "n01819313": "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita", 
        "n02002556": "white stork, Ciconia ciconia", 
        "n01667778": "terrapin", 
        "n01755581": "diamondback, diamondback rattlesnake, Crotalus adamanteus", 
        "n01924916": "flatworm, platyhelminth", 
        "n01751748": "sea snake", 
        "n01984695": "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish", 
        "n01729977": "green snake, grass snake", 
        "n01614925": "bald eagle, American eagle, Haliaeetus leucocephalus", 
        "n01608432": "kite", 
        "n01443537": "goldfish, Carassius auratus", 
        "n01770393": "scorpion", 
        "n01855672": "goose", 
        "n01560419": "bulbul", 
        "n01592084": "chickadee", 
        "n01914609": "sea anemone, anemone", 
        "n01582220": "magpie", 
        "n01667114": "mud turtle", 
        "n01985128": "crayfish, crawfish, crawdad, crawdaddy", 
        "n01820546": "lorikeet", 
        "n01773797": "garden spider, Aranea diademata", 
        "n02006656": "spoonbill", 
        "n01986214": "hermit crab", 
        "n01484850": "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias", 
        "n01749939": "green mamba", 
        "n01828970": "bee eater", 
        "n02018795": "bustard", 
        "n01695060": "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis", 
        "n01729322": "hognose snake, puff adder, sand viper", 
        "n01677366": "common iguana, iguana, Iguana iguana", 
        "n01734418": "king snake, kingsnake", 
        "n01843383": "toucan", 
        "n01806143": "peacock",
        "n01773549": "barn spider, Araneus cavaticus", 
        "n01775062": "wolf spider, hunting spider", 
        "n01728572": "thunder snake, worm snake, Carphophis amoenus", 
        "n01601694": "water ouzel, dipper", 
        "n01978287": "Dungeness crab, Cancer magister", 
        "n01930112": "nematode, nematode worm, roundworm", 
        "n01739381": "vine snake", 
        "n01883070": "wombat", 
        "n01774384": "black widow, Latrodectus mactans", 
        "n02037110": "oystercatcher, oyster catcher", 
        "n01795545": "black grouse", 
        "n02027492": "red-backed sandpiper, dunlin, Erolia alpina", 
        "n01531178": "goldfinch, Carduelis carduelis", 
        "n01944390": "snail", 
        "n01494475": "hammerhead, hammerhead shark", 
        "n01632458": "spotted salamander, Ambystoma maculatum", 
        "n01698640": "American alligator, Alligator mississipiensis", 
        "n01675722": "banded gecko", 
        "n01877812": "wallaby, brush kangaroo", 
        "n01622779": "great grey owl, great gray owl, Strix nebulosa", 
        "n01910747": "jellyfish", 
        "n01860187": "black swan, Cygnus atratus", 
        "n01796340": "ptarmigan", 
        "n01833805": "hummingbird", 
        "n01685808": "whiptail, whiptail lizard", 
        "n01756291": "sidewinder, horned rattlesnake, Crotalus cerastes", 
        "n01514859": "hen", 
        "n01753488": "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus", 
        "n02058221": "albatross, mollymawk", 
        "n01632777": "axolotl, mud puppy, Ambystoma mexicanum", 
        "n01644900": "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui", 
        "n02018207": "American coot, marsh hen, mud hen, water hen, Fulica americana", 
        "n01664065": "loggerhead, loggerhead turtle, Caretta caretta", 
        "n02028035": "redshank, Tringa totanus", 
        "n02012849": "crane", 
        "n01776313": "tick", 
        "n02077923": "sea lion", 
        "n01774750": "tarantula", 
        "n01742172": "boa constrictor, Constrictor constrictor", 
        "n01943899": "conch", 
        "n01798484": "prairie chicken, prairie grouse, prairie fowl", 
        "n02051845": "pelican", 
        "n01824575": "coucal", 
        "n02013706": "limpkin, Aramus pictus", 
        "n01955084": "chiton, coat-of-mail shell, sea cradle, polyplacophore", 
        "n01773157": "black and gold garden spider, Argiope aurantia", 
        "n01665541": "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea", 
        "n01498041": "stingray", "n01978455": "rock crab, Cancer irroratus", 
        "n01693334": "green lizard, Lacerta viridis", 
        "n01950731": "sea slug, nudibranch", 
        "n01829413": "hornbill", 
        "n01514668": "cock"
    }

    def __init__(
        self,
        root: Union[str, Path],
        splits: list = ["train"],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._dataset_root = (Path(root)/'imagenet-100')
        
        self._splits = splits

        self._samples = []
        for _split in self._splits:
            self._images_root = self._dataset_root/_split

            self.wnids, self.wnid_to_idx = find_classes(self._images_root)
            self.classes = [self._WNID_TO_CLASS[wnid] for wnid in self.wnids]
            self.class_to_idx = {
                class_name: idx for wnid, idx in self.wnid_to_idx.items() for class_name in self._WNID_TO_CLASS[wnid]
            }

            self._samples += make_dataset(self._images_root, self.wnid_to_idx, extensions=".jpeg")

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._samples)

def get_imagenet100(root, tfms_train, tfms_test):
    
    ds_train = Imagenet100(root=root,
                           splits=['train'],
                           transform=tfms_train)
    ds_valid = None
    
    ds_test  = Imagenet100(root=root,
                           splits=['val'],
                           transform=tfms_test)
    
    return ds_train, ds_test, ds_valid

def get_imagenet100_full(root, tfms_train, tfms_test):
    
    ds_train = Imagenet100(root=root,
                           splits=['train', 'val'],
                           transform=tfms_train)
    
    ds_valid = Imagenet100(root=root,
                           splits=['val'],
                           transform=tfms_test)

    ds_test = Imagenet100(root=root,
                          splits=['val'],
                          transform=tfms_test)

    return ds_train, ds_test, ds_valid

def get_imagenet100_full_eval(root, tfms_train, tfms_test):
    
    return get_imagenet100(root=root, tfms_train=tfms_train, tfms_test=tfms_test)