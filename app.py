import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
import gradio as gr

from model import SegHeadDPT

# ── Конфигурация ──────────────────────────────────────────────────────
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_H              = 224
IMG_W              = 1400
INTERMEDIATE_LAYERS = [3, 5, 8, 11]
WEIGHTS_DIR        = 'weights'
HEAD_PATH          = os.path.join(WEIGHTS_DIR, 'model_exp17.pt')
BB_PATH            = os.path.join(WEIGHTS_DIR, 'dinov2_exp17.pt')

YANDEX_LINKS = {
    HEAD_PATH: 'https://disk.yandex.ru/d/RS5DNE-W83j7rA',
    BB_PATH:   'https://disk.yandex.ru/d/TaKAGseYPGkExQ',
}

CLASS_COLORS = {
    1: (255, 50,  50,  180),
    2: (50,  200, 50,  180),
    3: (50,  100, 255, 180),
    4: (255, 200, 0,   180),
}
CLASS_NAMES = ['Фон', 'Дефект 1', 'Дефект 2', 'Дефект 3', 'Дефект 4']

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])


# ── Скачивание весов с Яндекс Диска ──────────────────────────────────
def download_from_yadisk(public_url, dest_path):
    print(f'Скачиваем {os.path.basename(dest_path)}...')
    api_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download'
    r = requests.get(api_url, params={'public_key': public_url})
    r.raise_for_status()
    download_url = r.json()['href']

    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f'\r  {downloaded/1e6:.1f} / {total/1e6:.1f} МБ', end='')
    print(f'\n  Сохранено: {dest_path}')


def ensure_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for path, url in YANDEX_LINKS.items():
        if not os.path.exists(path):
            download_from_yadisk(url, path)
        else:
            print(f'Веса найдены: {path}')


# ── Загрузка модели ───────────────────────────────────────────────────
def load_model():
    ensure_weights()
    print(f'Загружаем модель на {DEVICE}...')

    backbone = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vitb14',
        pretrained=False, verbose=False)
    backbone.load_state_dict(torch.load(BB_PATH, map_location=DEVICE))
    backbone = backbone.to(DEVICE).eval()

    head = SegHeadDPT().to(DEVICE)
    head.load_state_dict(torch.load(HEAD_PATH, map_location=DEVICE))
    head.eval()

    print('Модель загружена.')
    return backbone, head


backbone, head = load_model()


# ── Инференс ──────────────────────────────────────────────────────────
def preprocess(img_pil):
    img = img_pil.convert('RGB').resize((IMG_W, IMG_H), Image.BILINEAR)
    tensor = NORMALIZE(transforms.ToTensor()(img))
    return tensor.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict(img_tensor):
    if DEVICE == 'cuda':
        with torch.amp.autocast('cuda'):
            feats  = backbone.get_intermediate_layers(
                img_tensor, n=INTERMEDIATE_LAYERS, return_class_token=False)
            logits = head(feats)
    else:
        feats  = backbone.get_intermediate_layers(
            img_tensor, n=INTERMEDIATE_LAYERS, return_class_token=False)
        logits = head(feats)
    return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


# ── Визуализация ──────────────────────────────────────────────────────
def overlay_mask(img_pil, mask_small):
    w, h = img_pil.size
    mask_img = Image.fromarray(mask_small).resize((w, h), Image.NEAREST)
    mask_np  = np.array(mask_img)

    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    for cls_id, color in CLASS_COLORS.items():
        layer    = Image.new('RGBA', (w, h), color)
        mask_bin = Image.fromarray((mask_np == cls_id).astype(np.uint8) * 255)
        overlay.paste(layer, mask=mask_bin)

    result = img_pil.convert('RGBA')
    return Image.alpha_composite(result, overlay).convert('RGB')


# ── Основная функция ──────────────────────────────────────────────────
def segment(image):
    if image is None:
        return None, 'Загрузите изображение'

    img_pil    = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    img_model  = img_pil.convert('RGB').resize((IMG_W, IMG_H), Image.BILINEAR)
    img_tensor = preprocess(img_pil)
    mask       = predict(img_tensor)
    result_img = overlay_mask(img_model, mask)

    unique, counts = np.unique(mask, return_counts=True)
    total_px = mask.size
    lines = []
    for cls_id, cnt in zip(unique, counts):
        pct = cnt / total_px * 100
        if cls_id == 0:
            lines.append(f'Фон: {pct:.1f}%')
        else:
            lines.append(f'{CLASS_NAMES[cls_id]}: {pct:.2f}% площади')

    detected = [c for c in unique if c != 0]
    if not detected:
        lines.append('→ Дефекты не обнаружены')

    return result_img, '\n'.join(lines)


# ── Интерфейс ─────────────────────────────────────────────────────────
with gr.Blocks(title='Детекция дефектов стали') as demo:
    gr.Markdown("""
    # Детекция дефектов стального проката

    Загрузите изображение стальной поверхности.
    Модель выделит дефекты цветом:
    🔴 Дефект 1 &nbsp;&nbsp; 🟢 Дефект 2 &nbsp;&nbsp; 🔵 Дефект 3 &nbsp;&nbsp; 🟡 Дефект 4
    """)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(label='Исходное изображение', type='pil')
            btn = gr.Button('Найти дефекты', variant='primary')
        with gr.Column():
            out_img  = gr.Image(label='Результат сегментации')
            out_text = gr.Textbox(label='Статистика', lines=6)

    btn.click(fn=segment, inputs=inp, outputs=[out_img, out_text])


if __name__ == '__main__':
    demo.launch(share=False)
