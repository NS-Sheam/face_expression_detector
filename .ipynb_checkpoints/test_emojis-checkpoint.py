import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMOTION_EMOJIS = {
    'Angry': '😣', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😊',
    'Sad': '😢', 'Surprise': '😮', 'Neutral': '😐'
}

def draw_text_with_pil(frame, text, position, font_size=30, color=(0, 255, 0)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    font_paths = [
        ("./NotoColorEmoji.ttf", "Noto Color Emoji"),
        ("/System/Library/Fonts/Apple Color Emoji.ttc", "Apple Color Emoji"),
        ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "Arial Unicode")
    ]
    font = None
    font_name = None
    for path, name in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            font_name = name
            logger.info(f"Using font: {font_name}")
            break
        except IOError:
            continue
    if font is None:
        font = ImageFont.load_default()
        font_name = "Default"
        logger.warning("No emoji-compatible font found. Using default font (rendering text without emojis).")
        for emo, emoji in EMOTION_EMOJIS.items():
            text = text.replace(emoji, f"[{emo}]")
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def test_emojis():
    logger.info("Testing emoji rendering")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame.fill(255)
    y_offset = 50
    for emotion, emoji in EMOTION_EMOJIS.items():
        text = f"{emotion}: {emoji}"
        frame = draw_text_with_pil(frame, text, (50, y_offset), font_size=30, color=(0, 0, 255))
        y_offset += 40
    cv2.imwrite('emoji_test_output.png', frame)
    logger.info("Saved emoji test output to emoji_test_output.png")
    cv2.imshow('Emoji Test', frame)
    logger.info("Displaying emoji test window. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    logger.info("Emoji test completed")

if __name__ == "__main__":
    test_emojis()