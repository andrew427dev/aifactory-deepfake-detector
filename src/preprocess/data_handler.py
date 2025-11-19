import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, *, return_paths: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.return_paths = return_paths

        # real / fake 최상위 폴더
        self.real_dir = os.path.join(root_dir, "real")
        self.fake_dir = os.path.join(root_dir, "fake")

        # 최종 이미지 경로(상대경로)와 라벨(0=real, 1=fake)
        self.image_paths: list[str] = []
        self.labels: list[int] = []

        # real / fake 각각 재귀적으로 이미지 파일 수집
        self._collect_images(self.real_dir, label=0)
        self._collect_images(self.fake_dir, label=1)

    def _collect_images(self, base_dir: str, label: int) -> None:
        """
        base_dir(real 또는 fake) 아래의 모든 하위 디렉토리를 재귀적으로 돌며
        jpg/jpeg/png 파일만 수집해서 self.image_paths / self.labels에 저장한다.
        """
        for current_root, _, files in os.walk(base_dir):
            for fname in files:
                fname_lower = fname.lower()
                if not fname_lower.endswith((".jpg", ".jpeg", ".png")):
                    continue

                full_path = os.path.join(current_root, fname)
                # root_dir 기준 상대 경로로 저장해 두면 __getitem__에서 다시 조합하기 쉽다.
                rel_path = os.path.relpath(full_path, self.root_dir)

                self.image_paths.append(rel_path)
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 저장해 둔 상대 경로를 root_dir와 합쳐서 실제 파일 경로 생성
        img_rel_path = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_rel_path)

        # 이미지 로딩
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        # transform 적용
        if self.transform:
            image = self.transform(image)

        sample = (image, torch.tensor(label, dtype=torch.float32))

        # 필요하면 경로까지 같이 반환
        if self.return_paths:
            sample = (*sample, img_rel_path)

        return sample


def get_transforms(image_size):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def _build_loader(
    directory,
    transform,
    batch_size,
    shuffle=False,
    num_workers: int = 4,
):
    dataset = DeepfakeDataset(directory, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def get_dataloaders(
    train_dir,
    image_size,
    batch_size: int = 32,
    val_dir: str | None = None,
    test_dir: str | None = None,
    num_workers: int = 4,
):
    """
    train_dir / val_dir / test_dir 구조를 기반으로 DataLoader 3개를 반환한다.
    num_workers, pin_memory, persistent_workers 등을 한 곳에서 관리한다.
    """
    transform = get_transforms(image_size)

    if val_dir is None:
        # 단일 디렉토리에서 train/val/test split
        train_ratio = 0.7
        val_ratio = 0.15
        dataset = DeepfakeDataset(train_dir, transform=transform)
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        return train_loader, val_loader, test_loader

    # train / val / test 디렉토리가 따로 있는 경우
    train_loader = _build_loader(
        train_dir, transform, batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = _build_loader(
        val_dir, transform, batch_size, shuffle=False, num_workers=num_workers
    )
    test_directory = test_dir or val_dir
    test_loader = _build_loader(
        test_directory, transform, batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
