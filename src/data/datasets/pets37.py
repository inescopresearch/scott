import os
import os.path
import pathlib
from typing import Any, Callable, Optional, Tuple, Union
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

class Pets37(VisionDataset):
    
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``binary-category`` (int): Binary label for cat or dog.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )

    _VALID_TARGET_TYPES = ("category", "binary-category", "segmentation")

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        splits: list = ['trainval'],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
        self.splits_txt_file = 'list.txt' if len(splits) > 1 else f'{splits[0]}.txt'
        image_ids = []
        self._labels = []
        self._bin_labels = []
        with open(self._anns_folder / self.splits_txt_file) as file:
            for line in file:
                if line[0] != '#':
                    image_id, label, bin_label, _ = line.strip().split()
                    image_ids.append(image_id)
                    self._labels.append(int(label) - 1)
                    self._bin_labels.append(int(bin_label) - 1)

        self.bin_classes = ["Cat", "Dog"]
        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.bin_class_to_idx = dict(zip(self.bin_classes, range(len(self.bin_classes))))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]


    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")
        target = self._labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, target

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)

def get_pets37(root, tfms_train, tfms_test):
    
    ds_train = Pets37(root=root,
                      splits=['trainval'],
                      transform=tfms_train,
                      download=True)
    
    ds_valid = None

    ds_test = Pets37(root=root,
                      splits=['test'],
                      transform=tfms_test,
                      download=True)
    
    return ds_train, ds_test, ds_valid

def get_pets37_full(root, tfms_train, tfms_test):
    
    ds_train = Pets37(root=root,
                      splits=['trainval', 'test'],
                      transform=tfms_train,
                      download=True)
    
    ds_valid = None

    ds_test = Pets37(root=root,
                      splits=['test'],
                      transform=tfms_test,
                      download=True)
    
    return ds_train, ds_test, ds_valid


def get_pets37_full_eval(root, tfms_train, tfms_test):
    
    return get_pets37(root=root, tfms_train=tfms_train, tfms_test=tfms_test)