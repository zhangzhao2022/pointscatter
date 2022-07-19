from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from pointscatter.modeling.layers.coords import get_meshgrid
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize
import os
import numpy as np
from PIL import Image


@SEGMENTORS.register_module()
class EncoderDecoderVisual(EncoderDecoder):
    def __init__(self,
                 visual_dir: str = None,
                 label_dir: str = None,
                 img_suffix: str = '.png',
                 seg_map_suffix: str = '.png',
                 **kwargs):
        super(EncoderDecoderVisual, self).__init__(**kwargs)
        self.decode_head_name = self.decode_head.__class__.__name__
        self.visual_dir = visual_dir
        self.label_dir = label_dir
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.use_aux_in_train = True if hasattr(self, 'auxiliary_head') else False
        if self.use_aux_in_train:
            self.aux_head_name = self.auxiliary_head.__class__.__name__
            if isinstance(self.auxiliary_head, nn.ModuleList):
                self.aux_head_name = self.auxiliary_head[0].__class__.__name__

    def encode_decode(self, img, img_metas):
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def encode_auxiliary_decode(self, img, img_metas):
        x = self.extract_feat(img)
        out = self._auxiliary_head_forward_test(x, img_metas)
        return out

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        file_id = img_meta[0]['ori_filename'].replace(self.img_suffix, '')
        label_path = os.path.join(self.label_dir, file_id + self.seg_map_suffix)
        gt_label = Image.open(label_path)
        gt_label = np.array(gt_label)
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = img[:, :, y1:y2, x1:x2]
                if self.decode_head_name == 'PointScatterHead':
                    ps_head_pred = self.encode_decode(crop_img, img_meta)
                    crop_seg_logit = self.post_process(ps_head_pred)
                    preds[:, 1, :, :] += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                else:
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                    preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[3] - x2),
                                    int(y1), int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        if self.decode_head_name == 'PointScatterHead':
            preds[:, 0, :, :] = 1. - preds[:, 1, :, :]
            preds_prob = preds
        else:
            preds_prob = F.softmax(preds, dim=1)

        self.save_result(file_id, gt_label, preds_prob[0, 1, :, :])

        return preds

    def post_process(self, pred):
        scores, offsets, H, W, M = pred

        meshgrid = get_meshgrid((H, W), scores.device)
        meshgrid = meshgrid.reshape(H * W, 2)
        coord_pred = offsets + meshgrid.unsqueeze(1)
        output_coord = coord_pred * M

        scores = scores.squeeze(0).reshape(-1).sigmoid()
        output_coord = output_coord.squeeze(0).reshape(-1, 2)

        # convert coord to mask
        height, width = H * M, W * M
        coord_index = output_coord.floor().long()
        legal = coord_index.gt(-1).all(-1) & coord_index[:, 0].lt(height) & \
                coord_index[:, 1].lt(width) & (scores > 0.1)
        scores = scores[legal]
        coord_index = coord_index[legal]

        logits = torch.zeros((height, width), dtype=output_coord.dtype, device=output_coord.device)
        logits.index_put_((coord_index[:, 0], coord_index[:, 1]), scores)

        return logits

    def save_result(self, file_id, gt_label, prob_mask):
        gt_path = os.path.join(self.visual_dir, 'label')
        os.makedirs(gt_path, exist_ok=True)
        pred_path = os.path.join(self.visual_dir, 'prediction')
        os.makedirs(pred_path, exist_ok=True)
        prob_path = os.path.join(self.visual_dir, 'prob_mask')
        os.makedirs(prob_path, exist_ok=True)
        gt_path = os.path.join(gt_path, file_id + self.seg_map_suffix)
        pred_path = os.path.join(pred_path, file_id + self.seg_map_suffix)

        suffix_ = self.seg_map_suffix.split('.')[-1]
        prob_path = os.path.join(prob_path, file_id + self.seg_map_suffix.replace(suffix_, 'npz'))
        np.savez(prob_path, prob_mask=np.float32(prob_mask.cpu().numpy()))
        gt_label = Image.fromarray(np.uint8(gt_label * 255))
        gt_label.save(gt_path)
        pred_label_seg = Image.fromarray(np.uint8((prob_mask > 0.5).cpu().numpy() * 255))
        pred_label_seg.save(pred_path)

        return
