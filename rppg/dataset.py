import os
from glob import glob
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class rPPGFrameDataset(Dataset):
    """
    Frame-level dataset for rPPG.
    :param root_dir: Path to directory containing case directories.
    :param transform: Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self.case_dirs = [path for path in glob(os.path.join(root_dir, "*")) if os.path.isdir(path)]

        self.frame_paths = []
        self.hrs = []

        for case_dir in self.case_dirs:
            labels_path = os.path.join(case_dir, 'labels.csv')
            labels_df = pd.read_csv(labels_path)

            case_frame_paths = [os.path.join(case_dir, f'frame_{frame_id:05d}.png') for frame_id in labels_df['frame'].values]
            self.frame_paths.extend(case_frame_paths)

            case_hrs = labels_df['hr'].values
            self.hrs.extend(case_hrs)
        
        # Check that all frame paths are valid
        idxs_to_remove = [self.frame_paths.index(frame_path) for frame_path in self.frame_paths
                          if not os.path.isfile(frame_path)]
        
        for idx in sorted(idxs_to_remove, reverse=True):
            self.frame_paths.pop(idx)
            self.hrs.pop(idx)
            
        if len(idxs_to_remove) > 0:
            print(f"Removed {len(idxs_to_remove)} invalid frame paths")

        for frame_path in self.frame_paths:
            assert os.path.exists(frame_path), f"Frame path {frame_path} does not exist"
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        frame = Image.fromarray(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB))

        if self.transform:
            frame = self.transform(frame)
        
        hr = self.hrs[idx]
        hr = torch.tensor(hr, dtype=torch.float32)

        return frame, hr


class rPPGClipDataset(Dataset):
    """
    Clip-level dataset for rPPG.
    :param root_dir: Path to directory containing case directories.
    :param clip_length: Number of frames per clip.
    :param transform: Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir: str, clip_length: int, transform: callable=None) -> None:
        self.root_dir = root_dir
        self.clip_length = clip_length

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self.case_dirs = [path for path in glob(os.path.join(root_dir, "*")) if os.path.isdir(path)]

        self.labelled_clips = []
        
        for case_dir in self.case_dirs:
            labels_path = os.path.join(case_dir, 'labels.csv')
            labels_df = pd.read_csv(labels_path)
            frame_ids = labels_df['frame'].values
            num_frames = len(frame_ids)

            # Iterate in chunks of clip_length
            for start_idx in range(0, num_frames - clip_length + 1, clip_length):
                end_idx = start_idx + clip_length

                clip_frame_ids = frame_ids[start_idx:end_idx]
                clip_frame_paths = [os.path.join(case_dir, f'frame_{frame_id:05d}.png') for frame_id in clip_frame_ids]

                clip_hrs = labels_df.iloc[start_idx:end_idx]['hr'].values

                self.labelled_clips.append((clip_frame_paths, clip_hrs))
            
    def __len__(self):
        return len(self.labelled_clips)
    
    def __getitem__(self, idx):
        clip_frame_paths, clip_hrs = self.labelled_clips[idx]

        clip_frames = [Image.fromarray(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
                       for path in clip_frame_paths]

        if self.transform:
            clip_frames = [self.transform(img) for img in clip_frames]
        
        clip_frames = torch.stack(clip_frames)
        clip_frames = clip_frames.permute(1, 0, 2, 3)  # (channels, clip_length, height, width)
        clip_hrs = torch.tensor(clip_hrs, dtype=torch.float32)

        return clip_frames, clip_hrs
