import torch
from skimage.transform import resize
import numpy as np


def process_image(numpy_image, new_shape):
    h, w, _ = numpy_image.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    input_img = np.pad(numpy_image, pad, 'constant', constant_values=127.5) / 255.
    input_img = resize(input_img, (new_shape, new_shape, 3), mode='reflect')
    input_img = np.transpose(input_img, (2, 0, 1))
    return np.expand_dims(input_img, axis=0).astype(np.float32)


@torch.jit.script
def bbox_iou(box1, box2, x1y1x2y2: bool = True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

@torch.jit.script
def non_max_suppression(prediction, num_classes: int, conf_thres: float = 0.5, nms_thres: float = 0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = torch.zeros(prediction.shape, device=prediction.device).to(dtype=prediction.dtype)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]


    # Filter out confidence scores below threshold
    image_pred = prediction[0]
    conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
    image_pred = image_pred[conf_mask]
    # If none are remaining => process next image
    output = torch.zeros(0, device=prediction.device)
    if image_pred.size(0) != 0:
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        
        # Get the detections with the particular class
        detections_class = detections[detections[:, -1] == 0]  # people
        # Sort the detections by maximum objectness confidence
        _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
        detections_class = detections_class[conf_sort_index]
        # Perform non-maximum suppression
        max_detections = []
        should_break = False
        while detections_class.size(0) != 0 and not should_break:
            # Get detection with highest confidence and save as max detection
            max_detections.append(detections_class[0].unsqueeze(0))
            # Stop if we're at the last detection
            if detections_class.size(0) == 1:
                should_break = True
            else:
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]
        if max_detections != []:
            output = torch.cat(max_detections)
    return output

