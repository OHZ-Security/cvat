# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import onnxruntime as ort


class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="model.onnx")
        self.labels = labels

    def load_network(self, model):
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(
                model, providers=["CPUExecutionProvider"], sess_options=so
            )
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        # Resize keeping aspect ratio, then pad to new_shape, centred
        # (same as Ultralytics' LetterBox with auto=False)
        shape = im.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def _infer(self, image: np.ndarray):
        # image is RGB, HWC, uint8
        image, ratio, dwdh = self.letterbox(image)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        im = np.ascontiguousarray(image).astype(np.float32) / 255

        # end2end export: [1, 300, 6] rows of x1, y1, x2, y2, score, class
        detections = self.model.run(self.output_details, {self.input_details[0]: im})[0][0]

        boxes = detections[:, :4]
        scores = detections[:, 4]
        labels = detections[:, 5].astype(np.int32)

        boxes -= np.array(dwdh * 2)
        boxes /= ratio
        return boxes.round().astype(np.int32), labels, scores

    def infer(self, image, threshold):
        image = np.array(image)
        h, w, _ = image.shape
        boxes, labels, scores = self._infer(image)

        results = []
        for label, score, box in zip(labels, scores, boxes):
            if score >= threshold:
                xtl = max(int(box[0]), 0)
                ytl = max(int(box[1]), 0)
                xbr = min(int(box[2]), w)
                ybr = min(int(box[3]), h)

                results.append(
                    {
                        "confidence": str(score),
                        "label": self.labels.get(int(label), "unknown"),
                        "points": [xtl, ytl, xbr, ybr],
                        "type": "rectangle",
                    }
                )

        return results
