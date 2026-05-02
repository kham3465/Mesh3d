"""Download CASIA-WebFace from huggingface and save as folder structure:
dataset_root/<identity>/<image>.jpg

This script requires `datasets` (pip install datasets).
"""
import os
from pathlib import Path


def main(dataset_root='data/casia_webface'):
    from datasets import load_dataset
    ds = load_dataset('SaffalPoosh/casia_web_face')
    os.makedirs(dataset_root, exist_ok=True)
    cnt, created_dirs = 0, set()

    for split in ds:
        print(f"Processing split: {split}")
        for ex in ds[split]:
            # Ưu tiên lấy 'label' vì đây là cấu trúc chuẩn của bộ CASIA trên HuggingFace
            label = ex.get('label')
            identity = str(label) if label is not None else str(ex.get('identity') or ex.get('id') or 'unknown')
            
            img = ex.get('image')
            if img is None:
                img = ex.get('img')
            if img is None:
                continue

            label_dir = Path(dataset_root) / identity
            if identity not in created_dirs:
                label_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.add(identity)

            fn = ex.get('image_name') or ex.get('img_name') or f'{cnt}.jpg'
            fp = label_dir / fn
            try:
                if hasattr(img, 'save'):
                    img.save(fp)
                else:
                    with open(fp, 'wb') as f:
                        f.write(img)
                cnt += 1
                if cnt % 10000 == 0:
                    print(f"Saved {cnt} images...")
            except Exception as e:
                print('skip', e)

    print(f"Done! Saved {cnt} images to {os.path.abspath(dataset_root)}")
    print(f"Total classes found: {len(os.listdir(dataset_root))}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='data/casia_webface')
    args = parser.parse_args()
    main(args.dataset_root)
