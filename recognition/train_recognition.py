import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from recognition.models import EmbeddingNet
from recognition.dataset import default_transforms
from recognition.losses import ArcMarginProduct

def train_one_epoch(model, classifier, loader, optimizer, device, backbone_finetune):
    model.train()
    classifier.train()
    total, loss_sum = 0, 0.0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc='Training', leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        if backbone_finetune:
            emb = model(imgs)
        else:
            with torch.no_grad():
                emb = model(imgs)
        logits = classifier(emb, labels)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    return loss_sum / max(total, 1)

def eval_one_epoch(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    total, correct = 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc='Evaluating', leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            emb = model(imgs)
            # During eval, we don't apply margin penalty, just cosine similarity
            if isinstance(classifier, ArcMarginProduct):
                logits = torch.matmul(F.normalize(emb), F.normalize(classifier.weight).t()) * classifier.s
            else:
                logits = classifier(emb)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True, help='train folder (ImageFolder structure)')
    p.add_argument('--val_root', default=None, help='optional val folder')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--embedding_size', type=int, default=512)
    p.add_argument('--backbone', default='resnet50')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--fine_tune', action='store_true', help='allow backbone weights to update if set')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save_dir', default='checkpoints')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    device = torch.device(args.device)
    transform = default_transforms(112)

    train_dataset = datasets.ImageFolder(args.data_root, transform=transform)
    classes = train_dataset.classes
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = None
    if args.val_root:
        val_dataset = datasets.ImageFolder(args.val_root, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = len(classes)
    os.makedirs(args.save_dir, exist_ok=True)

    model = EmbeddingNet(embedding_size=args.embedding_size, backbone=args.backbone, pretrained=args.pretrained)
    model.to(device)

    # Use ArcMarginProduct instead of nn.Linear for better recognition accuracy
    classifier = ArcMarginProduct(
        in_features=args.embedding_size, 
        out_features=num_classes, 
        s=30.0, m=0.5).to(device)

    # choose params to optimize
    if args.fine_tune:
        params = list(model.parameters()) + list(classifier.parameters())
    else:
        # freeze backbone
        for p in model.backbone.parameters():
            p.requires_grad = False
        params = list(classifier.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, classifier, train_loader, optimizer, device, args.fine_tune)
        scheduler.step()
        val_acc = None
        if val_loader is not None:
            val_acc = eval_one_epoch(model, classifier, val_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc
        # save checkpoint (full)
        ck = {
            'model': model.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'classes': classes,
            'backbone': args.backbone,
            'embedding_size': args.embedding_size,
        }
        
        # Save latest
        latest_fn = os.path.join(args.save_dir, 'latest_recog_ck.pth')
        torch.save(ck, latest_fn)

        # Save best
        if val_acc is not None and val_acc >= best_acc:
            best_fn = os.path.join(args.save_dir, 'best_recog_ck.pth')
            torch.save(ck, best_fn)

        elapsed = time.time() - t0
        if val_acc is not None:
            print(f'Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s')
        else:
            print(f'Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} time={elapsed:.1f}s')

    print('Training finished')