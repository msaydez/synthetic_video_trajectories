import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math
from ssm import MambaTrackletPredictor

class SlopeTrackDataset(Dataset):
    def __init__(self, full_tracks, seq_len=5):
        self.seq_len = seq_len
        self.track_dict = {}
        for track in full_tracks:
            tid = track['track_id']
            if tid not in self.track_dict:
                self.track_dict[tid] = []
            self.track_dict[tid].extend(track['frames'])
        self.valid_ids = [tid for tid in self.track_dict if len(self.track_dict[tid]) > seq_len]

    def sample_sequence(self, frames):
        frames = sorted(frames, key=lambda x: x[0])
        start = random.randint(0, len(frames) - self.seq_len - 1)
        selected = frames[start:start + self.seq_len + 1]
        bbox_seq = torch.tensor([item[1] for item in selected[:-1]])
        gt_next_bbox = torch.tensor(selected[-1][1])
        return bbox_seq, gt_next_bbox

    def __getitem__(self, idx):
        anchor_id = self.valid_ids[idx % len(self.valid_ids)]
        anchor_frames = self.track_dict[anchor_id]
        anchor_seq, anchor_next = self.sample_sequence(anchor_frames)

        pos_seq, _ = self.sample_sequence(anchor_frames)
        negative_ids = [tid for tid in self.valid_ids if tid != anchor_id]
        neg_id = random.choice(negative_ids)
        neg_frames = self.track_dict[neg_id]
        neg_seq, _ = self.sample_sequence(neg_frames)

        return anchor_seq, anchor_next, pos_seq, neg_seq

    def __len__(self):
        return len(self.valid_ids)


def collate_fn(batch):
    anchor_seq, anchor_next, pos_seq, neg_seq = zip(*batch)
    return (
        torch.stack(anchor_seq),
        torch.stack(anchor_next),
        torch.stack(pos_seq),
        torch.stack(neg_seq),
    )


# ====================================
# DIOU & CIOU Losses
# ====================================

def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(pred, gt, eps=1e-7):
    pred = box_cxcywh_to_xyxy(pred)
    gt = box_cxcywh_to_xyxy(gt)
    x1 = torch.max(pred[..., 0], gt[..., 0])
    y1 = torch.max(pred[..., 1], gt[..., 1])
    x2 = torch.min(pred[..., 2], gt[..., 2])
    y2 = torch.min(pred[..., 3], gt[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    gt_area = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
    union = pred_area + gt_area - inter + eps
    iou = inter / union
    return iou, inter, union, pred, gt


def compute_diou_loss(pred, gt):
    iou, _, _, pred_xy, gt_xy = compute_iou(pred, gt)
    center_pred = (pred_xy[..., :2] + pred_xy[..., 2:]) / 2
    center_gt = (gt_xy[..., :2] + gt_xy[..., 2:]) / 2
    center_dist = (center_pred - center_gt).pow(2).sum(-1)
    enclose_x1 = torch.min(pred_xy[..., 0], gt_xy[..., 0])
    enclose_y1 = torch.min(pred_xy[..., 1], gt_xy[..., 1])
    enclose_x2 = torch.max(pred_xy[..., 2], gt_xy[..., 2])
    enclose_y2 = torch.max(pred_xy[..., 3], gt_xy[..., 3])
    c = ((enclose_x2 - enclose_x1).pow(2) + (enclose_y2 - enclose_y1).pow(2)) + 1e-7
    diou = 1 - iou + center_dist / c
    return diou.mean()


def compute_ciou_loss(pred, gt):
    iou, _, _, pred_xy, gt_xy = compute_iou(pred, gt)
    center_pred = (pred_xy[..., :2] + pred_xy[..., 2:]) / 2
    center_gt = (gt_xy[..., :2] + gt_xy[..., 2:]) / 2
    center_dist = (center_pred - center_gt).pow(2).sum(-1)
    enclose_x1 = torch.min(pred_xy[..., 0], gt_xy[..., 0])
    enclose_y1 = torch.min(pred_xy[..., 1], gt_xy[..., 1])
    enclose_x2 = torch.max(pred_xy[..., 2], gt_xy[..., 2])
    enclose_y2 = torch.max(pred_xy[..., 3], gt_xy[..., 3])
    c = ((enclose_x2 - enclose_x1).pow(2) + (enclose_y2 - enclose_y1).pow(2)) + 1e-7
    w_pred = pred_xy[..., 2] - pred_xy[..., 0]
    h_pred = pred_xy[..., 3] - pred_xy[..., 1]
    w_gt = gt_xy[..., 2] - gt_xy[..., 0]
    h_gt = gt_xy[..., 3] - gt_xy[..., 1]
    v = (4 / (math.pi ** 2)) * (torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    ciou = 1 - iou + center_dist / c + alpha * v
    return ciou.mean()


# ====================================
# Training Loop
# ====================================

def train(train_tracks, val_tracks):
    train_set = SlopeTrackDataset(train_tracks)
    val_set = SlopeTrackDataset(val_tracks)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = MambaTrackletPredictor().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(500):
        model.train()
        total_train_loss = 0.0

        for anchor_seq, anchor_next, pos_seq, neg_seq in train_loader:
            anchor_seq = anchor_seq.cuda()
            anchor_next = anchor_next.cuda()
            pos_seq = pos_seq.cuda()
            neg_seq = neg_seq.cuda()

            delta_anchor, embed_anchor = model(anchor_seq)
            pred_bbox = anchor_seq[:, -1, :] + delta_anchor

            _, embed_pos = model(pos_seq)
            _, embed_neg = model(neg_seq)

            diou = compute_diou_loss(pred_bbox, anchor_next)
            ciou = compute_ciou_loss(pred_bbox, anchor_next)
            motion_loss = diou + ciou

            pos_sim = F.cosine_similarity(embed_anchor, embed_pos)
            neg_sim = F.cosine_similarity(embed_anchor, embed_neg)

            pos_loss = (1 - pos_sim).mean()
            neg_loss = F.relu(neg_sim - 0.5).mean()
            contrastive_loss = pos_loss + neg_loss

            loss = motion_loss + 0.5 * contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/500] - Loss: {avg_loss:.6f}")

        # Validation after each epoch
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for anchor_seq, anchor_next, pos_seq, neg_seq in val_loader:
                anchor_seq = anchor_seq.cuda()
                anchor_next = anchor_next.cuda()
                pos_seq = pos_seq.cuda()
                neg_seq = neg_seq.cuda()

                delta_anchor, embed_anchor = model(anchor_seq)
                pred_bbox = anchor_seq[:, -1, :] + delta_anchor

                _, embed_pos = model(pos_seq)
                _, embed_neg = model(neg_seq)

                diou = compute_diou_loss(pred_bbox, anchor_next)
                ciou = compute_ciou_loss(pred_bbox, anchor_next)
                motion_loss = diou + ciou

                pos_sim = F.cosine_similarity(embed_anchor, embed_pos)
                neg_sim = F.cosine_similarity(embed_anchor, embed_neg)

                pos_loss = (1 - pos_sim).mean()
                neg_loss = F.relu(neg_sim - 0.5).mean()
                contrastive_loss = pos_loss + neg_loss

                loss = motion_loss + 0.5 * contrastive_loss

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/500] - Val Loss: {avg_val_loss:.6f}")

import pickle

with open("train_set.pkl", "rb") as f:
    train_tracks = pickle.load(f)

with open("val_set.pkl", "rb") as f:
    val_tracks = pickle.load(f)

train(train_tracks,val_tracks)