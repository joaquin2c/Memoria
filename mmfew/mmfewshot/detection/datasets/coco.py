# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS

from mmfewshot.detection.core import eval_map
from mmdet.datasets.coco import CocoDataset
from terminaltables import AsciiTable

from .base import BaseFewShotDataset

# pre-defined classes split for few shot setting

COCO_SPLIT1 = dict(
    ALL_CLASSES=('bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'traffic light', 'fire hydrant',
                 'stop sign', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella',
                 'suitcase','baseball bat', 'skateboard',
                 'tennis racket', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'banana', 'apple',
                 'sandwich', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'toothbrush'),
    NOVEL_CLASSES=('bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'bird', 'elephant', 'dog', 'horse', 'sheep',
                   'cow', 'chair', 'couch', 'potted plant',
                   'dining table', 'tv'),
    BASE_CLASSES=('truck', 'cat','traffic light', 'fire hydrant', 'stop sign',
                  'bench', 'bear', 'zebra',
                  'giraffe', 'backpack', 'umbrella',
                  'suitcase', 'baseball bat', 'skateboard',
                  'tennis racket', 'wine glass', 'cup', 'fork',
                  'knife', 'spoon', 'banana', 'apple', 'sandwich',
                  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                  'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'toothbrush'))
COCO_SPLIT2 = dict(
    ALL_CLASSES=('bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'traffic light', 'fire hydrant',
                 'stop sign', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella',
                 'suitcase','baseball bat', 'skateboard',
                 'tennis racket', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'banana', 'apple',
                 'sandwich', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'toothbrush'),
    NOVEL_CLASSES=('toothbrush', 'chair', 'baseball bat', 'laptop', 'clock',
                   'bench', 'remote', 'airplane', 'dog', 'sheep', 'bear',
                   'fire hydrant', 'backpack', 'scissors', 'giraffe',
                   'bus', 'bird'),
    BASE_CLASSES=('bicycle', 'car', 'motorcycle','train', 'truck',
                 'traffic light','stop sign', 'cat','horse', 'cow',
                 'elephant', 'zebra','umbrella','suitcase', 'skateboard',
                 'tennis racket', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'banana', 'apple',
                 'sandwich', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'mouse',
                 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'book', 'vase','teddy bear'))

COCO_SPLIT3 = dict(
    ALL_CLASSES=('bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'traffic light', 'fire hydrant',
                 'stop sign', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella',
                 'suitcase','baseball bat', 'skateboard',
                 'tennis racket', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'banana', 'apple',
                 'sandwich', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'toothbrush'),
    NOVEL_CLASSES=('bicycle', 'remote', 'zebra', 'elephant', 'teddy bear',
                  'oven','cake', 'knife', 'tv', 'toaster','toilet', 'sink',
                  'fork', 'backpack', 'cat', 'fire hydrant', 'banana'),
    BASE_CLASSES=('car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'traffic light',
                 'stop sign', 'bench', 'bird', 'dog',
                 'horse', 'sheep', 'cow', 'bear','giraffe', 'umbrella',
                 'suitcase','baseball bat', 'skateboard',
                 'tennis racket', 'wine glass', 'cup','spoon','apple',
                 'sandwich', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'laptop', 'mouse',
                 'keyboard', 'cell phone', 'microwave',
                 'book', 'clock', 'vase','scissors', 'toothbrush'))

Original_COCO={'person': 1,'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9,
               'traffic light': 10,  'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17,
               'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
               'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37,
               'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44,
               'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54,
               'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63,
               'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75,
               'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84,
               'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}


@DATASETS.register_module()
class FewShotCocoDataset(BaseFewShotDataset, CocoDataset):
    """COCO dataset for few shot detection.

    Args:
        classes (str | Sequence[str] | None): Classes for model training and
            provide fixed label for each class. When classes is string,
            it will load pre-defined classes in :obj:`FewShotCocoDataset`.
            For example: 'BASE_CLASSES', 'NOVEL_CLASSES` or `ALL_CLASSES`.
        num_novel_shots (int | None): Max number of instances used for each
            novel class. If is None, all annotation will be used.
            Default: None.
        num_base_shots (int | None): Max number of instances used for each base
            class. If is None, all annotation will be used. Default: None.
        ann_shot_filter (dict | None): Used to specify the class and the
            corresponding maximum number of instances when loading
            the annotation file. For example: {'dog': 10, 'person': 5}.
            If set it as None, `ann_shot_filter` will be
            created according to `num_novel_shots` and `num_base_shots`.
        min_bbox_area (int | float | None):  Filter images with bbox whose
            area smaller `min_bbox_area`. If set to None, skip
            this filter. Default: None.
        dataset_name (str | None): Name of dataset to display. For example:
            'train dataset' or 'query dataset'. Default: None.
        test_mode (bool): If set True, annotation will not be loaded.
            Default: False.
    """

    def __init__(self,
                 classes: Optional[Union[str, Sequence[str]]] = None,
                 num_novel_shots: Optional[int] = None,
                 num_base_shots: Optional[int] = None,
                 ann_shot_filter: Optional[Dict[str, int]] = None,
                 min_bbox_area: Optional[Union[int, float]] = None,
                 dataset_name: Optional[str] = None,
                 test_mode: bool = False,
                 coordinate_offset: List[int] = [-1, -1, 0, 0],
                 split="Split1",
                 **kwargs) -> None:
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name

        if split=="Split2":
            self.SPLIT = COCO_SPLIT2
        elif split=="Split3":
            self.SPLIT = COCO_SPLIT3
        else:
            self.SPLIT = COCO_SPLIT1

        assert classes is not None, f'{self.dataset_name}: classes in ' \
                                    f'`FewShotCocoDataset` can not be None.'
        # `ann_shot_filter` will be used to filter out excess annotations
        # for few shot setting. It can be configured manually or generated
        # by the `num_novel_shots` and `num_base_shots`
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area = min_bbox_area
        self.CLASSES = self.get_classes(classes)
        if ann_shot_filter is None:
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'

        # these values would be set in `self.load_annotations_coco`
        self.cat_ids = []
        self.cat2label = {}
        self.coco = None
        self.img_ids = None
        self.coordinate_offset = coordinate_offset
        super().__init__(
            classes=None,
            ann_shot_filter=ann_shot_filter,
            dataset_name=dataset_name,
            test_mode=test_mode,
            **kwargs)

    def get_classes(self, classes: Union[str, Sequence[str]]) -> List[str]:
        """Get class names.

        It supports to load pre-defined classes splits.
        The pre-defined classes splits are:
        ['ALL_CLASSES', 'NOVEL_CLASSES', 'BASE_CLASSES']

        Args:
            classes (str | Sequence[str]): Classes for model training and
                provide fixed label for each class. When classes is string,
                it will load pre-defined classes in `FewShotCocoDataset`.
                For example: 'NOVEL_CLASSES'.

        Returns:
            list[str]: list of class names.
        """
        # configure few shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name} : not a pre-defined classes or split ' \
               f'in COCO_SPLIT.'
            class_names = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:
                assert self.num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not have ' \
                    f'base instances.'
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def _create_ann_shot_filter(self) -> Dict:
        """Generate `ann_shot_filter` for novel and base classes.

        Returns:
            dict[str, int]: The number of shots to keep for each class.
        """
        ann_shot_filter = {}
        # generate annotation filter for novel classes
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT['NOVEL_CLASSES']:
                ann_shot_filter[class_name] = self.num_novel_shots
        # generate annotation filter for base classes
        if self.num_base_shots is not None:
            for class_name in self.SPLIT['BASE_CLASSES']:
                ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter

    def load_annotations(self, ann_cfg: List[Dict]) -> List[Dict]:
        """Support to Load annotation from two type of ann_cfg.

            - type of 'ann_file': COCO-style annotation file.
            - type of 'saved_dataset': Saved COCO dataset json.

        Args:
            ann_cfg (list[dict]): Config of annotations.

        Returns:
            list[dict]: Annotation infos.
        """
        data_infos = []
        for ann_cfg_ in ann_cfg:
            if ann_cfg_['type'] == 'saved_dataset':
                data_infos += self.load_annotations_saved(ann_cfg_['ann_file'])
            elif ann_cfg_['type'] == 'ann_file':
                data_infos += self.load_annotations_coco(ann_cfg_['ann_file'])
            else:
                raise ValueError(f'not support annotation type '
                                 f'{ann_cfg_["type"]} in ann_cfg.')
        return data_infos

    def load_annotations_coco(self, ann_file: str) -> List[Dict]:
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        # to keep the label order equal to the order in CLASSES
        if len(self.cat_ids) == 0:
            for i, class_name in enumerate(self.CLASSES):
                cat_id = self.coco.get_cat_ids(cat_names=[class_name])[0]
                self.cat_ids.append(cat_id)
                self.cat2label[cat_id] = i
        else:
            # check categories id consistency between different files
            for i, class_name in enumerate(self.CLASSES):
                cat_id = self.coco.get_cat_ids(cat_names=[class_name])[0]
                assert self.cat2label[cat_id] == i, \
                    'please make sure all the json files use same ' \
                    'categories id for same class'
        self.img_ids = self.coco.get_img_ids()

        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            info['ann'] = self._get_ann_info(info)
            # to support different version of coco since some annotation file
            # contain images from train2014 and val2014 at the same time
            if 'train2014' in info['filename']:
                info['filename'] = 'train2014/' + info['filename']
            elif 'val2014' in info['filename']:
                info['filename'] = 'val2014/' + info['filename']
            elif 'instances_val2017' in ann_file:
                info['filename'] = 'val2017/' + info['filename']
            elif 'instances_train2017' in ann_file:
                info['filename'] = 'train2017/' + info['filename']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids
        ), f'{self.dataset_name}: Annotation ids in {ann_file} are not unique!'
        return data_infos

    def _get_ann_info(self, data_info: Dict) -> Dict:
        """Get COCO annotation by index.

        Args:
            data_info(dict): Data info.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = data_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(data_info, ann_info)

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Overwrite the function in CocoDataset.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int64).tolist()

    def _filter_imgs(self,
                     min_size: int = 32,
                     min_bbox_area: Optional[int] = None) -> List[int]:
        """Filter images that do not meet the requirements.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area (int | None): Filter images with bbox whose
                area smaller `min_bbox_area`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indices of data_infos.
        """
        valid_inds = []
        valid_img_ids = []
        if min_bbox_area is None:
            min_bbox_area = self.min_bbox_area
        for i, img_info in enumerate(self.data_infos):
            # filter empty image
            if self.filter_empty_gt and img_info['ann']['labels'].size == 0:
                continue
            # filter images smaller than `min_size`
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            # filter image with bbox smaller than min_bbox_area
            # it is usually used in Attention RPN
            if min_bbox_area is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
            valid_img_ids.append(img_info['id'])
        # update coco img_ids
        self.img_ids = valid_img_ids
        return valid_inds

    def evaluate(self,
                 results: List[Sequence],
                 metric: Union[str, List[str]] = 'mAP',
                 logger: Optional[object] = None,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thr: Optional[Union[float, Sequence[float]]] = 0.5,
                 class_splits: Optional[List[str]] = None) -> Dict:
        """Evaluation in VOC protocol and summary results of different splits
        of classes.

        Args:
            results (list[list | tuple]): Predictions of the model.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'. Default: mAP.
            logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            class_splits: (list[str] | None): Calculate metric of classes
                split  defined in VOC_SPLIT. For example:
                ['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'].
                Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        # It should be noted that in the original mmdet implementation,
        # the four coordinates are reduced by 1 when the annotation
        # is parsed. Here we following detectron2, only xmin and ymin
        # will be reduced by 1 during training. The groundtruth used for
        # evaluation or testing keep consistent with original xml
        # annotation file and the xmin and ymin of prediction results
        # will add 1 for inverse of data loading logic.
        for i in range(len(results)):
            for j in range(len(results[i])):
                for k in range(4):
                    results[i][j][:, k] -= self.coordinate_offset[k]

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        if class_splits is not None:
            for k in class_splits:
                assert k in self.SPLIT.keys(), 'undefiend classes split.'
            class_splits = {k: self.SPLIT[k] for k in class_splits}
            class_splits_mean_aps = {k: [] for k in class_splits.keys()}

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, ap_results = eval_map(
                    results,
                    annotations,
                    classes=self.CLASSES,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset='voc07',
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

                # calculate evaluate results of different class splits
                if class_splits is not None:
                    for k in class_splits.keys():
                        aps = [
                            cls_results['ap']
                            for i, cls_results in enumerate(ap_results)
                            if self.CLASSES[i] in class_splits[k]
                        ]
                        class_splits_mean_ap = np.array(aps).mean().item()
                        class_splits_mean_aps[k].append(class_splits_mean_ap)
                        eval_results[
                            f'{k}: AP{int(iou_thr * 100):02d}'] = round(
                                class_splits_mean_ap, 3)

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            if class_splits is not None:
                for k in class_splits.keys():
                    mAP = sum(class_splits_mean_aps[k]) / len(
                        class_splits_mean_aps[k])
                    print_log(f'{k} mAP: {mAP}', logger=logger)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results



@DATASETS.register_module()
class DrawCocoDataset(BaseFewShotDataset):
    """COCO dataset for few shot detection.

    Args:
        classes (str | Sequence[str] | None): Classes for model training and
            provide fixed label for each class. When classes is string,
            it will load pre-defined classes in :obj:`FewShotCocoDataset`.
            For example: 'BASE_CLASSES', 'NOVEL_CLASSES` or `ALL_CLASSES`.
        num_novel_shots (int | None): Max number of instances used for each
            novel class. If is None, all annotation will be used.
            Default: None.
        num_base_shots (int | None): Max number of instances used for each base
            class. If is None, all annotation will be used. Default: None.
        ann_shot_filter (dict | None): Used to specify the class and the
            corresponding maximum number of instances when loading
            the annotation file. For example: {'dog': 10, 'person': 5}.
            If set it as None, `ann_shot_filter` will be
            created according to `num_novel_shots` and `num_base_shots`.
        min_bbox_area (int | float | None):  Filter images with bbox whose
            area smaller `min_bbox_area`. If set to None, skip
            this filter. Default: None.
        dataset_name (str | None): Name of dataset to display. For example:
            'train dataset' or 'query dataset'. Default: None.
        test_mode (bool): If set True, annotation will not be loaded.
            Default: False.
    """

    def __init__(self,
                 classes: Optional[Union[str, Sequence[str]]] = None,
                 num_novel_shots: Optional[int] = None,
                 num_base_shots: Optional[int] = None,
                 ann_shot_filter: Optional[Dict[str, int]] = None,
                 min_bbox_area: Optional[Union[int, float]] = None,
                 dataset_name: Optional[str] = None,
                 test_mode: bool = False,
                 split="Split1",
                 coordinate_offset: List[int] = [-1, -1, 0, 0],
                 **kwargs) -> None:
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name

        if split=="Split2":
            self.SPLIT = COCO_SPLIT2
        elif split=="Split3":
            self.SPLIT = COCO_SPLIT3
        else:
            self.SPLIT = COCO_SPLIT1

        self.original=Original_COCO
        assert classes is not None, f'{self.dataset_name}: classes in ' \
                                    f'`FewShotCocoDataset` can not be None.'
        # `ann_shot_filter` will be used to filter out excess annotations
        # for few shot setting. It can be configured manually or generated
        # by the `num_novel_shots` and `num_base_shots`
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area = min_bbox_area
        self.CLASSES = self.get_classes(classes)
        if ann_shot_filter is None:
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'

        # these values would be set in `self.load_annotations_coco`
        self.cat_ids = []
        self.cat2label = {}
        self.coco = None
        self.img_ids = None
        self.coordinate_offset = coordinate_offset
        super().__init__(
            classes=None,
            ann_shot_filter=ann_shot_filter,
            dataset_name=dataset_name,
            test_mode=test_mode,
            **kwargs)

    def get_classes(self, classes: Union[str, Sequence[str]]) -> List[str]:
        """Get class names.

        It supports to load pre-defined classes splits.
        The pre-defined classes splits are:
        ['ALL_CLASSES', 'NOVEL_CLASSES', 'BASE_CLASSES']

        Args:
            classes (str | Sequence[str]): Classes for model training and
                provide fixed label for each class. When classes is string,
                it will load pre-defined classes in `FewShotCocoDataset`.
                For example: 'NOVEL_CLASSES'.

        Returns:
            list[str]: list of class names.
        """
        # configure few shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name} : not a pre-defined classes or split ' \
               f'in COCO_SPLIT.'
            class_names = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:
                assert self.num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not have ' \
                    f'base instances.'
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def _create_ann_shot_filter(self) -> Dict:
        """Generate `ann_shot_filter` for novel and base classes.

        Returns:
            dict[str, int]: The number of shots to keep for each class.
        """
        ann_shot_filter = {}
        # generate annotation filter for novel classes
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT['NOVEL_CLASSES']:
                ann_shot_filter[class_name] = self.num_novel_shots
        # generate annotation filter for base classes
        if self.num_base_shots is not None:
            for class_name in self.SPLIT['BASE_CLASSES']:
                ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter

    def getCatIdsCOCO(self,original,catNms=[], supNms=[], catIds=[]):
        catNms = catNms if self._isArrayLike(catNms) else [catNms]
        supNms = supNms if self._isArrayLike(supNms) else [supNms]
        catIds = catIds if self._isArrayLike(catIds) else [catIds] 
        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = original
        else:
            cats = original
            cats = cats if len(catNms) == 0 else {cat:cats.get(cat) for cat in catNms}
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
            ids = cats
        return ids

    def _isArrayLike(self,obj):
        return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

    def load_annotations(self, ann_file: List[Dict]) -> List[Dict]:
        """Support to Load annotation from two type of ann_cfg.

            - type of 'ann_file': COCO-style annotation file.
            - type of 'saved_dataset': Saved COCO dataset json.

        Args:
            ann_cfg (list[dict]): Config of annotations.

        Returns:
            list[dict]: Annotation infos.
        """
        if len(self.cat_ids) == 0:
            for i, class_name in enumerate(self.CLASSES):
                cat_id = self.getCatIdsCOCO(self.original,catNms=[class_name])
                key, value=list(cat_id.items())[0]
                self.cat_ids.append(value)
                self.cat2label[key] = i
        else:
            # check categories id consistency between different files
            for i, class_name in enumerate(self.CLASSES):
                cat_id = self.get_cat_ids(self.original,cat_names=[class_name])
                key=list(cat_id.keys())[0]
                assert self.cat2label[key] == i, \
                    'please make sure all the json files use same ' \
                    'categories id for same class'

        data_infos = []
        for ann_cfg in ann_file:
            if ann_cfg['type'] == 'saved_dataset':
                data_infos += self.load_annotations_saved(ann_cfg['ann_file'])
            elif ann_cfg['type'] == 'ann_file':
                ann_classes = ann_cfg.get('ann_classes', None)
                if ann_classes is not None:
                    for c in ann_classes:
                        assert c in self.CLASSES, \
                            f'{self.dataset_name}: ann_classes must in ' \
                            f'dataset classes.'
                else:
                    ann_classes = self.CLASSES
                data_infos += self.load_annotations_xml(
                    ann_cfg['ann_file'], ann_classes)
            else:
                raise ValueError(f'not support annotation type '
                                 f'{ann_cfg_["type"]} in ann_cfg.')
        return data_infos

    def load_annotations_xml(
            self,
            ann_file: str,
            classes: Optional[List[str]] = None) -> List[Dict]:
        """Load annotation from XML style ann_file.

        It supports using image id or image path as image names
        to load the annotation file.

        Args:
            ann_file (str): Path of annotation file.
            classes (list[str] | None): Specific classes to load form xml file.
                If set to None, it will use classes of whole dataset.
                Default: None.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        classes = mmcv.list_from_file(ann_file)
        for clase in classes:
            path_class=osp.join(self.img_prefix,clase)
            img_names=mmcv.list_from_file(path_class)
            for img_name in img_names:
                # ann file in image path format
                if 'COCO' in ann_file:
                    data_infos+=self.load_annotations_support(clase,img_name,ann_file)
                    continue
                else:
                    raise ValueError('Cannot infer dataset year from img_prefix')

        return data_infos


    def load_annotations_support(
            self,
            clase:str,
            img_name:str,
            ann_file:str) ->List[Dict]:

        support_data=[]
        all_paths = clase.split("/")
        data_clase=all_paths[-2]
        filename=f'{data_clase}/{img_name}.jpg'
        bboxes = np.array([[0,0,640,480]])
        labels = np.array([self.cat2label[data_clase]])
        bboxes_ignore=np.array([])
        labels_ignore=np.array([])

        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))

        support_data.append(dict(
                    id=img_name,
                    filename=filename,
                    width=640,
                    height=480,
                    ann=ann_info))
        return support_data


    

    

    def _get_ann_info(self, data_info: Dict) -> Dict:
        """Get COCO annotation by index.

        Args:
            data_info(dict): Data info.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = data_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(data_info, ann_info)

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Overwrite the function in CocoDataset.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int64).tolist()

    def _filter_imgs(self,
                     min_size: int = 32,
                     min_bbox_area: Optional[int] = None) -> List[int]:
        """Filter images that do not meet the requirements.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area (int | None): Filter images with bbox whose
                area smaller `min_bbox_area`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indices of data_infos.
        """
        valid_inds = []
        valid_img_ids = []
        if min_bbox_area is None:
            min_bbox_area = self.min_bbox_area
        for i, img_info in enumerate(self.data_infos):
            # filter empty image
            if self.filter_empty_gt and img_info['ann']['labels'].size == 0:
                continue
            # filter images smaller than `min_size`
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            # filter image with bbox smaller than min_bbox_area
            # it is usually used in Attention RPN
            if min_bbox_area is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
            valid_img_ids.append(img_info['id'])
        # update coco img_ids
        self.img_ids = valid_img_ids
        return valid_inds
    
    def evaluate(self,
                 results: List[Sequence],
                 metric: Union[str, List[str]] = 'mAP',
                 logger: Optional[object] = None,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thr: Optional[Union[float, Sequence[float]]] = 0.5,
                 class_splits: Optional[List[str]] = None) -> Dict:
        """Evaluation in VOC protocol and summary results of different splits
        of classes.

        Args:
            results (list[list | tuple]): Predictions of the model.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'. Default: mAP.
            logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            class_splits: (list[str] | None): Calculate metric of classes
                split  defined in VOC_SPLIT. For example:
                ['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'].
                Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        # It should be noted that in the original mmdet implementation,
        # the four coordinates are reduced by 1 when the annotation
        # is parsed. Here we following detectron2, only xmin and ymin
        # will be reduced by 1 during training. The groundtruth used for
        # evaluation or testing keep consistent with original xml
        # annotation file and the xmin and ymin of prediction results
        # will add 1 for inverse of data loading logic.
        for i in range(len(results)):
            for j in range(len(results[i])):
                for k in range(4):
                    results[i][j][:, k] -= self.coordinate_offset[k]

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        if class_splits is not None:
            for k in class_splits:
                assert k in self.SPLIT.keys(), 'undefiend classes split.'
            class_splits = {k: self.SPLIT[k] for k in class_splits}
            class_splits_mean_aps = {k: [] for k in class_splits.keys()}

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, ap_results = eval_map(
                    results,
                    annotations,
                    classes=self.CLASSES,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset='voc07',
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

                # calculate evaluate results of different class splits
                if class_splits is not None:
                    for k in class_splits.keys():
                        aps = [
                            cls_results['ap']
                            for i, cls_results in enumerate(ap_results)
                            if self.CLASSES[i] in class_splits[k]
                        ]
                        class_splits_mean_ap = np.array(aps).mean().item()
                        class_splits_mean_aps[k].append(class_splits_mean_ap)
                        eval_results[
                            f'{k}: AP{int(iou_thr * 100):02d}'] = round(
                                class_splits_mean_ap, 3)

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            if class_splits is not None:
                for k in class_splits.keys():
                    mAP = sum(class_splits_mean_aps[k]) / len(
                        class_splits_mean_aps[k])
                    print_log(f'{k} mAP: {mAP}', logger=logger)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

        
        

        
@DATASETS.register_module()
class FewShotCocoCopyDataset(FewShotCocoDataset):
    """Copy other COCO few shot datasets' `data_infos` directly.

    This dataset is mainly used for model initialization in some meta-learning
    detectors. In their cases, the support data are randomly sampled
    during training phase and they also need to be used in model
    initialization before evaluation. To copy the random sampling results,
    this dataset supports to load `data_infos` of other datasets via `ann_cfg`

    Args:
        ann_cfg (list[dict] | dict): contain `data_infos` from other
            dataset. Example: [dict(data_infos=FewShotCocoDataset.data_infos)]
    """

    def __init__(self, ann_cfg: Union[List[Dict], Dict], **kwargs) -> None:
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: Union[List[Dict], Dict]) -> List[Dict]:
        """Parse annotation config from a copy of other dataset's `data_infos`.

        Args:
            ann_cfg (list[dict] | dict): contain `data_infos` from other
                dataset. Example:
                [dict(data_infos=FewShotCocoDataset.data_infos)]

        Returns:
            list[dict]: Annotation information.
        """
        data_infos = []
        if isinstance(ann_cfg, dict):
            assert ann_cfg.get('data_infos', None) is not None, \
                'ann_cfg of FewShotCocoCopyDataset require data_infos.'
            # directly copy data_info
            data_infos = ann_cfg['data_infos']
        elif isinstance(ann_cfg, list):
            for ann_cfg_ in ann_cfg:
                assert ann_cfg_.get('data_infos', None) is not None, \
                    'ann_cfg of FewShotCocoCopyDataset require data_infos.'
                # directly copy data_info
                data_infos += ann_cfg_['data_infos']
        return data_infos


@DATASETS.register_module()
class FewShotCocoDefaultDataset(FewShotCocoDataset):
    """FewShot COCO Dataset with some pre-defined annotation paths.

    :obj:`FewShotCocoDefaultDataset` provides pre-defined annotation files
    to ensure the reproducibility. The pre-defined annotation files provide
    fixed training data to avoid random sampling. The usage of `ann_cfg' is
    different from :obj:`FewShotCocoDataset`. The `ann_cfg' should contain
    two filed: `method` and `setting`.

    Args:
        ann_cfg (list[dict]): Each dict should contain
            `method` and `setting` to get corresponding
            annotation from `DEFAULT_ANN_CONFIG`.
            For example: [dict(method='TFA', setting='1shot')].
    """
    coco_benchmark = {
        f'{shot}SHOT': [
            dict(
                type='ann_file',
                ann_file=f'data/few_shot_ann/coco/benchmark_{shot}shot/'
                f'full_box_{shot}shot_{class_name}_trainval.json')
            for class_name in COCO_SPLIT2['ALL_CLASSES']
        ]
        for shot in [10, 30]
    }

    # pre-defined annotation config for model reproducibility
    DEFAULT_ANN_CONFIG = dict(
        TFA=coco_benchmark,
        FSCE=coco_benchmark,
        Attention_RPN={
            **coco_benchmark, 'Official_10SHOT': [
                dict(
                    type='ann_file',
                    ann_file='data/few_shot_ann/coco/attention_rpn_10shot/'
                    'official_10_shot_from_instances_train2017.json')
            ]
        },
        MPSR=coco_benchmark,
        MetaRCNN=coco_benchmark,
        FSDetView=coco_benchmark)

    def __init__(self, ann_cfg: List[Dict], **kwargs) -> None:
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: List[Dict]) -> List[Dict]:
        """Parse pre-defined annotation config to annotation information.

        Args:
            ann_cfg (list[dict]): Each dict should contain
                `method` and `setting` to get corresponding
                annotation from `DEFAULT_ANN_CONFIG`.
                For example: [dict(method='TFA', setting='1shot')]

        Returns:
            list[dict]: Annotation information.
        """
        new_ann_cfg = []
        for ann_cfg_ in ann_cfg:
            assert isinstance(ann_cfg_, dict), \
                f'{self.dataset_name} : ann_cfg should be list of dict.'
            method = ann_cfg_['method']
            setting = ann_cfg_['setting']
            default_ann_cfg = self.DEFAULT_ANN_CONFIG[method][setting]
            ann_root = ann_cfg_.get('ann_root', None)
            if ann_root is not None:
                for i in range(len(default_ann_cfg)):
                    default_ann_cfg[i]['ann_file'] = \
                        osp.join(ann_root, default_ann_cfg[i]['ann_file'])
            new_ann_cfg += default_ann_cfg
        return super(FewShotCocoDataset, self).ann_cfg_parser(new_ann_cfg)
