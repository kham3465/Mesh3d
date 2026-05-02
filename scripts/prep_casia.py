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
    cnt = 0
    for split in ds:
        for ex in ds[split]:
            identity = str(ex.get('identity') or ex.get('id') or 'unknown')
            img = ex.get('image')
            if img is None:
                # some versions store 'img' or 'image_bytes'
                img = ex.get('img')
            if img is None:
                continue
            label_dir = Path(dataset_root) / identity
            label_dir.mkdir(parents=True, exist_ok=True)
            fn = ex.get('image_name') or ex.get('img_name') or f'{cnt}.jpg'
            fp = label_dir / fn
            try:
                # img may be PIL Image or bytes
                if hasattr(img, 'save'):
                    img.save(fp)
                else:
                    # assume bytes
                    with open(fp, 'wb') as f:
                        f.write(img)
                cnt += 1
            except Exception as e:
                print('skip', e)
    print('Saved images to', dataset_root)


if __name__ == '__main__':
    main()
