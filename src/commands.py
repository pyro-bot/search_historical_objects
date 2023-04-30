from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Dict, List, Tuple, Union
import click
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import xmltodict
import torchvision as tv
import torch
from PIL import Image as PILImage, ImageDraw


class Labels(Enum):
    city = 'поселения'
    point = 'курган'
    
    @classmethod
    def parse(cls, value):
        match value:
            case 'поселения':
                return cls.city
            case 'курган':
                return cls.point
    
    def index(self):
        return 0 if self == Labels.city else 1

    def __len__(self) -> int:
        return 2


@dataclass
class SegmentRectangle:
    x: int
    y: int
    xe: int
    ye: int
    label: Labels

    def get_min_x(self):
        return self.x

    def get_min_y(self):
        return self.y

    def get_max_x(self):
        return self.xe
    
    def get_max_y(self):
        return self.ye
    
@dataclass
class SegmentElipse:
    cx: int
    cy: int
    rx: int
    ry: int
    label: Labels

    @property
    def radius(self):
        return math.sqrt((self.rx)**2 + (self.ry)**2)

    def get_min_x(self):
        return max(0, self.cx - self.radius)

    def get_min_y(self):
        return max(0, self.cy - self.radius)

    def get_max_x(self):
        return self.cx + self.radius
    
    def get_max_y(self):
        return self.cy + self.radius

    @property
    def w(self):
        return abs(self.cx - self.rx)
    
    @property
    def h(self):
        return abs(self.cy - self.ry)
    

@dataclass
class SegmentPolygon:
    points: List[Tuple[int, int]]
    label: Labels

    
    def get_min_x(self):
        return get_min(0, self.points)


    def get_min_y(self):
        return get_min(1, self.points)

    def get_max_x(self):
        return get_max(0, self.points)
    
    def get_max_y(self):
        return get_max(1, self.points)


@dataclass
class Image:
    file: Path
    segments: List[Union[SegmentElipse, SegmentPolygon, SegmentRectangle]]


# @click.command('cutting')
# @click.option('--file', type=Path, required=True)
# @click.option('--dataset', type=Path, required=True)
def __photo_cutting(file: Path, dataset: Path, out_folder="dataset_out", temp_dir='temp', size=448, step=30):
    meta = xmltodict.parse(file.read_text('utf8'))['annotations']
    meta_images = meta['image']
    images: Dict[str, Image] = {}
    idx = 0
    while idx < len(meta_images):
        m = meta_images[idx]
        img = images.get(m['@name'])
        if img is None:
            img = Image(file=dataset / m['@name'], segments=[])
            images[m['@name']] = img
        match m:
            case {'polygon': poly}:
                img.segments.append(SegmentPolygon(
                    [tuple(list(map(float, ps.split(',')))) for ps in poly['@points'].split(';')],
                    poly['@label'],
                ))
                del m['polygon']
            case {'ellipse': elipses}:
                match elipses:
                    case list():
                        for elipses in elipses:
                            img.segments.append(SegmentElipse(
                                cx=float(elipses['@cx']), cy=float(elipses['@cy']),
                                rx=float(elipses['@rx']), ry=float(elipses['@ry']),
                                label=Labels.parse(elipses['@label'])
                            ))
                    case _:
                        img.segments.append(SegmentElipse(
                            cx=float(elipses['@cx']), cy=float(elipses['@cy']),
                            rx=float(elipses['@rx']), ry=float(elipses['@ry']),
                            label=Labels.parse(elipses['@label'])
                        ))
                del m['ellipse']
            case {'box': recangle}:
                img.segments.append(SegmentRectangle(
                    float(recangle['@xtl']), float(recangle['@ytl']),
                    float(recangle['@xbr']), float(recangle['@ybr']),
                    Labels.parse(recangle['@label'])
                ))
                del m['box']
            case _:
                idx += 1
    
    for obj in tqdm(images.values()):
            img = read_image(obj.file)
            mask = [PILImage.fromarray(torch.zeros(img.shape[1], img.shape[2], dtype=torch.uint8).numpy()) for _ in range(3)] # надо что бы число осей было 3
            for seg in obj.segments:
                match seg:
                    case SegmentRectangle():
                        ImageDraw.Draw(mask[seg.label.index()]).rectangle((seg.x, seg.y, seg.xe, seg.ye), fill=1, outline=1)
                    case SegmentPolygon():
                        ImageDraw.Draw(mask[seg.label.index()]).polygon(seg.points, outline=1, fill=1)
                    case SegmentElipse():
                        ImageDraw.Draw(mask[seg.label.index()]).ellipse(((seg.cx, seg.cy), (seg.cx+seg.rx, seg.cy+seg.ry)), fill=1, outline=1)
            mask = [np.array(o) for o in mask]
            out_mask = np.zeros((len(mask), mask[0].shape[0], mask[0].shape[1]), dtype='uint8')
            for i, m in enumerate(mask):
                out_mask[i, :, :] = m*255
            out_mask = torch.from_numpy(out_mask)
            to_file = dataset / temp_dir / obj.file.with_suffix(".jpg")
            to_file.parent.mkdir(parents=True, exist_ok=True)
            tv.io.write_jpeg(out_mask, str(to_file))


    datasetfiles = []
    for obj in tqdm(images.values()):
        for seg in obj.segments:
            start_x = max(0, int(seg.get_min_x()) - int(1*size))
            
            start_y = max(0, int(seg.get_min_y()) - int(1*size))
            end_x = int(seg.get_max_x()) + int(1*size)
            end_y = int(seg.get_max_y()) + int(1*size)
            file =  obj.file
            img = read_image(file=file)
            mask = read_image(file=dataset / temp_dir / file.with_suffix(".jpg"))
            # img = np.array(img)
            for ix in tqdm(range(start_x, end_x - size, step), leave=False):
                for iy in tqdm(range(start_y, end_y - size, step), leave=False):
                    segment = img[:, ix:ix+size, iy:iy+size]
                    segment_mask = mask[:, ix:ix+size, iy:iy+size]
                    to_file = (file.parent / out_folder / file.stem / f'{ix}_{iy}.jpg')
                    to_file.parent.mkdir(parents=True, exist_ok=True)
                    datasetfiles.append(str(to_file))
                    mask_file = to_file.with_stem(f'{to_file.stem}_mask')

                    tv.io.write_jpeg(segment, str(to_file.absolute()), quality=100)
                    tv.io.write_jpeg(segment_mask, str(mask_file.absolute()), quality=100)
    (dataset / 'dataset_out.txt').write_text(json.dumps(datasetfiles))

        

def get_min(cords, list):
    min = 0
    for item in list:
        match cords:
            case int():
                new_item = item[cords]
            case str():    
                new_item = getattr(item, cords)
        if new_item < min:
            min = new_item
    return min

def get_max(cords, list):
    max = -math.inf
    for item in list:
        match cords:
            case int():
                new_item = item[cords]
            case str():    
                new_item = getattr(item, cords)
        if new_item > max:
            max = new_item
    return max


def read_image(file: Path) -> torch.Tensor:
    img = PILImage.open(str(file.absolute()))
    match file.suffix:
        case '.tif':
            img = img.convert('RGB')
            img = np.array(img)
            img = torch.from_numpy(img).swapaxes(0, -1)
        case _:
            img = np.array(img)
            img = torch.from_numpy(img).swapaxes(0, -1)
    return img

if __name__ == '__main__':
    __photo_cutting(file=Path(r'datasets\annotations.xml'), dataset=Path('datasets/Участок Левобережного'))
    
