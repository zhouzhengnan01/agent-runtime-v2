from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

_ACL_INIT_LOCK = threading.Lock()
_ACL_INITIALIZED = False


def _ensure_acl_initialized(acl) -> None:
    """
    `acl.init()` must be called once per process.

    On some CANN builds, calling `acl.init()` again returns a non-zero code
    (observed: 100002) even though the runtime is already usable. We treat that
    as success if basic runtime calls work.
    """

    global _ACL_INITIALIZED
    if _ACL_INITIALIZED:
        return

    with _ACL_INIT_LOCK:
        if _ACL_INITIALIZED:
            return

        ret = acl.init()
        if ret != 0:
            # If ACL runtime was already initialized elsewhere, `acl.init()` may
            # return a non-zero "repeat initialize" code. Validate by calling a
            # simple runtime API.
            try:
                _, rt_ret = acl.rt.get_device_count()
            except Exception:
                rt_ret = None

            if rt_ret == 0 or int(ret) == 100002:
                _ACL_INITIALIZED = True
                return

            raise RuntimeError(f"acl.init failed: {ret}")

        _ACL_INITIALIZED = True


_COCO80_NAMES: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def _dtype_from_acl(acl_dtype: int) -> np.dtype:
    """
    Minimal dtype mapping for common OM models.

    Observed on 910B CANN:
      - input dtype 0 => float32
      - output dtype 1 => float16
    """

    if acl_dtype == 0:
        return np.float32
    if acl_dtype == 1:
        return np.float16
    # fallback (still allows memcpy); caller may cast later
    return np.float32


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        iou = _box_iou_one_to_many(boxes[i], boxes[order[1:]])
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def _letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    import cv2  # local import to keep module import light

    shape = img.shape[:2]  # h, w
    if shape[0] == 0 or shape[1] == 0:
        raise ValueError("invalid image shape")

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w, h
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0

    if (shape[1], shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


@dataclass(frozen=True)
class OmInputSpec:
    n: int
    c: int
    h: int
    w: int
    dtype: np.dtype


@dataclass(frozen=True)
class OmOutputSpec:
    shape: Tuple[int, ...]
    dtype: np.dtype


class AscendOmModel:
    """
    Minimal ACL OM runner (single model, static shapes).

    Notes:
    - This is intentionally lightweight and only supports common cases used by agent-v3.
    - `acl` is imported lazily; if container isn't Ascend-ready, init will fail clearly.
    """

    def __init__(self, model_path: str, *, device_id: int = 0) -> None:
        self.model_path = model_path
        self.device_id = int(device_id)

        try:
            import acl  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Ascend OM 推理需要 CANN Python 包 `acl`；请在 910B 机器上挂载 /usr/local/Ascend 并设置 PYTHONPATH"
            ) from e

        self._acl = acl

        _ensure_acl_initialized(acl)

        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            raise RuntimeError(f"acl.rt.set_device({self.device_id}) failed: {ret}")

        self._context, ret = acl.rt.create_context(self.device_id)
        if ret != 0:
            raise RuntimeError(f"acl.rt.create_context failed: {ret}")

        self._stream, ret = acl.rt.create_stream()
        if ret != 0:
            raise RuntimeError(f"acl.rt.create_stream failed: {ret}")

        self._model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.load_from_file failed: {ret}")

        self._desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self._desc, self._model_id)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.get_desc failed: {ret}")

        # input spec
        input_dims, ret = acl.mdl.get_input_dims(self._desc, 0)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.get_input_dims failed: {ret}")

        dims = list(input_dims.get("dims") or [])
        if len(dims) != 4:
            raise RuntimeError(f"unsupported input dims: {dims}")

        in_dtype = _dtype_from_acl(int(acl.mdl.get_input_data_type(self._desc, 0)))
        self.input_spec = OmInputSpec(n=int(dims[0]), c=int(dims[1]), h=int(dims[2]), w=int(dims[3]), dtype=in_dtype)

        # output spec (assume 1 output)
        output_dims, ret = acl.mdl.get_output_dims(self._desc, 0)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.get_output_dims failed: {ret}")

        out_shape = tuple(int(x) for x in (output_dims.get("dims") or []))
        if not out_shape:
            raise RuntimeError("empty output shape")

        out_dtype = _dtype_from_acl(int(acl.mdl.get_output_data_type(self._desc, 0)))
        self.output_spec = OmOutputSpec(shape=out_shape, dtype=out_dtype)

        self._input_size = int(acl.mdl.get_input_size_by_index(self._desc, 0))
        self._output_size = int(acl.mdl.get_output_size_by_index(self._desc, 0))

        # Allocate device buffers + datasets
        self._input_dev_ptr, ret = acl.rt.malloc(self._input_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc(input) failed: {ret}")

        self._output_dev_ptr, ret = acl.rt.malloc(self._output_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc(output) failed: {ret}")

        self._input_data_buffer = acl.create_data_buffer(self._input_dev_ptr, self._input_size)
        self._output_data_buffer = acl.create_data_buffer(self._output_dev_ptr, self._output_size)

        self._input_dataset = acl.mdl.create_dataset()
        _, ret = acl.mdl.add_dataset_buffer(self._input_dataset, self._input_data_buffer)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.add_dataset_buffer(input) failed: {ret}")

        self._output_dataset = acl.mdl.create_dataset()
        _, ret = acl.mdl.add_dataset_buffer(self._output_dataset, self._output_data_buffer)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.add_dataset_buffer(output) failed: {ret}")

        # Host output buffer
        self._output_host = np.empty((self._output_size,), dtype=np.uint8)

    def close(self) -> None:
        acl = self._acl
        try:
            try:
                acl.mdl.destroy_dataset(self._input_dataset)
            except Exception:
                pass
            try:
                acl.mdl.destroy_dataset(self._output_dataset)
            except Exception:
                pass
            try:
                acl.destroy_data_buffer(self._input_data_buffer)
            except Exception:
                pass
            try:
                acl.destroy_data_buffer(self._output_data_buffer)
            except Exception:
                pass
            try:
                acl.rt.free(self._input_dev_ptr)
            except Exception:
                pass
            try:
                acl.rt.free(self._output_dev_ptr)
            except Exception:
                pass
            try:
                acl.mdl.destroy_desc(self._desc)
            except Exception:
                pass
            try:
                acl.mdl.unload(self._model_id)
            except Exception:
                pass
            try:
                acl.rt.destroy_stream(self._stream)
            except Exception:
                pass
            try:
                acl.rt.destroy_context(self._context)
            except Exception:
                pass
            # NOTE: do not call acl.finalize() here; app may load multiple models.
        except Exception:
            pass

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        input_tensor: NCHW float32, shape matches input_spec.
        returns: output tensor, reshaped to output_spec.shape
        """

        acl = self._acl
        if input_tensor.dtype != self.input_spec.dtype:
            input_tensor = input_tensor.astype(self.input_spec.dtype)
        input_tensor = np.ascontiguousarray(input_tensor)

        if input_tensor.nbytes != self._input_size:
            raise ValueError(f"input nbytes mismatch: {input_tensor.nbytes} != {self._input_size}")

        ret = acl.rt.memcpy(self._input_dev_ptr, self._input_size, int(input_tensor.ctypes.data), self._input_size, 1)
        if ret != 0:
            raise RuntimeError(f"acl.rt.memcpy(H2D) failed: {ret}")

        ret = acl.mdl.execute(self._model_id, self._input_dataset, self._output_dataset)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.execute failed: {ret}")

        ret = acl.rt.memcpy(int(self._output_host.ctypes.data), self._output_size, self._output_dev_ptr, self._output_size, 2)
        if ret != 0:
            raise RuntimeError(f"acl.rt.memcpy(D2H) failed: {ret}")

        out = self._output_host.view(self.output_spec.dtype)
        return out.reshape(self.output_spec.shape)


class AscendOmYoloV8Detector:
    def __init__(self, model_path: str, *, device_id: Optional[int] = None) -> None:
        if device_id is None:
            device_id = int(os.getenv("ASCEND_DEVICE_ID", "0") or 0)
        self._om = AscendOmModel(model_path, device_id=int(device_id))
        self.input_h = self._om.input_spec.h
        self.input_w = self._om.input_spec.w

    def close(self) -> None:
        self._om.close()

    def predict_one(
        self,
        image_path: str,
        *,
        conf: float,
        iou: float,
        classes: Optional[Sequence[int]] = None,
        max_det: int = 300,
    ) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          - original width, original height
          - boxes_xyxy (N,4) in original pixel coords
          - scores (N,)
          - cls (N,)
        """

        import cv2  # local import

        img0 = cv2.imread(image_path)
        if img0 is None:
            raise RuntimeError(f"cv2.imread failed: {image_path}")
        h0, w0 = img0.shape[:2]

        img, r, (dw, dh) = _letterbox(img0, (self.input_h, self.input_w))
        img = img[:, :, ::-1]  # BGR->RGB
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[None, ...]  # NCHW

        pred = self._om.infer(img)  # (1,84,8400) fp16
        pred = pred.astype(np.float32, copy=False)

        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        # (84, 8400) -> (8400, 84)
        if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
            pred = pred.T

        if pred.ndim != 2 or pred.shape[1] < 6:
            raise RuntimeError(f"unexpected model output shape: {pred.shape}")

        # YOLOv8: (num, 84) = 4 + 80
        if pred.shape[1] == 84:
            boxes = pred[:, :4]
            cls_scores = pred[:, 4:]
            cls = cls_scores.argmax(axis=1)
            scores = cls_scores.max(axis=1)
        # YOLOv5 style: 4 + obj + 80
        elif pred.shape[1] == 85:
            boxes = pred[:, :4]
            obj = pred[:, 4]
            cls_scores = pred[:, 5:]
            cls = cls_scores.argmax(axis=1)
            scores = obj * cls_scores.max(axis=1)
        # already NMS output: x1,y1,x2,y2,score,cls
        elif pred.shape[1] == 6:
            det = pred
            scores = det[:, 4]
            cls = det[:, 5].astype(np.int32)
            boxes_xyxy = det[:, :4]
            keep = scores > float(conf)
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]
            cls = cls[keep]
            return w0, h0, boxes_xyxy, scores, cls
        else:
            raise RuntimeError(f"unsupported output features: {pred.shape[1]}")

        keep = scores > float(conf)
        boxes = boxes[keep]
        scores = scores[keep]
        cls = cls[keep].astype(np.int32)

        if boxes.size == 0:
            return w0, h0, boxes.reshape(0, 4), scores.reshape(0), cls.reshape(0)

        # If model outputs normalized boxes, upscale to pixels
        if float(np.nanmax(boxes)) <= 1.5:
            boxes = boxes * float(self.input_w)

        boxes_xyxy = _xywh_to_xyxy(boxes)

        if classes:
            allow = set(int(x) for x in classes)
            mask = np.array([int(c) in allow for c in cls], dtype=bool)
            boxes_xyxy = boxes_xyxy[mask]
            scores = scores[mask]
            cls = cls[mask]

        if boxes_xyxy.size == 0:
            return w0, h0, boxes_xyxy.reshape(0, 4), scores.reshape(0), cls.reshape(0)

        # Class-aware NMS via large offsets
        max_wh = 4096.0
        boxes_for_nms = boxes_xyxy + cls.reshape(-1, 1) * max_wh
        keep_idx = _nms(boxes_for_nms, scores, float(iou), int(max_det))

        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]
        cls = cls[keep_idx]

        # scale back to original image
        boxes_xyxy[:, [0, 2]] -= float(dw)
        boxes_xyxy[:, [1, 3]] -= float(dh)
        boxes_xyxy[:, :4] /= float(r + 1e-9)

        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w0)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w0)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h0)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h0)

        return w0, h0, boxes_xyxy, scores, cls

    @staticmethod
    def label_name(cls_id: int) -> str:
        if 0 <= int(cls_id) < len(_COCO80_NAMES):
            return _COCO80_NAMES[int(cls_id)]
        return str(int(cls_id))
