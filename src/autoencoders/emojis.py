import numpy as np
from PIL import Image

emoji_chars = ['😀', '😄', '😁', '😆', '😅', '🤣', '😂', '🙂', '🙃', '🫠', '😉', '😊', '😇', '🥰', '🤩', '😗', '😚', '🥲', '😋', '😛', '😜', '🤪', '😝', '🤑', '🤗', '🤭', '🫢', '🫣', '🤫', '🤔', '🫡', '🤐', '🤨', '😐', '😑', '😶', '😏', '😒', '🙄', '😬', '😮‍💨', '🤥', '😌', '😔', '😪', '🤤', '😴', '😷', '🤒', '🤕', '🤢', '🤮', '🤧', '🥵', '🥶', '🥴', '😵', '😵‍💫', '🤯', '🤠', '🥳', '🥸', '😎', '🤓', '🧐', '😕', '🫤', '😟', '🙁', '☹️', '☹', '😮', '😯', '😲', '😳', '🥺', '🥹', '😦', '😧', '😨', '😰', '😥', '😢', '😭', '😱', '😖', '😣', '😞', '😓', '😩', '😫', '🥱', '😤', '😡', '😠', '😈', '👿']

emoji_names = [
    "grinning face",
    "grinning face with smiling eyes",
    "beaming face with smiling eyes",
    "grinning squinting face",
    "grinning face with sweat",
    "rolling on the floor laughing",
    "face with tears of joy",
    "slightly smiling face",
    "upside-down face",
    "melting face",
    "winking face",
    "smiling face with smiling eyes",
    "smiling face with halo",
    "smiling face with hearts",
    "star-struck",
    "kissing face",
    "kissing face with closed eyes",
    "smiling face with tear",
    "face savoring food",
    "face with tongue",
    "winking face with tongue",
    "zany face",
    "squinting face with tongue",
    "money-mouth face",
    "smiling face with open hands",
    "face with hand over mouth",
    "face with open eyes and hand over mouth",
    "face with peeking eye",
    "shushing face",
    "thinking face",
    "saluting face",
    "zipper-mouth face",
    "face with raised eyebrow",
    "neutral face",
    "expressionless face",
    "face without mouth",
    "smirking face",
    "unamused face",
    "face with rolling eyes",
    "grimacing face",
    "face exhaling",
    "lying face",
    "relieved face",
    "pensive face",
    "sleepy face",
    "drooling face",
    "sleeping face",
    "face with medical mask",
    "face with thermometer",
    "face with head-bandage",
    "nauseated face",
    "face vomiting",
    "sneezing face",
    "hot face",
    "cold face",
    "woozy face",
    "face with crossed-out eyes",
    "face with spiral eyes",
    "exploding head",
    "cowboy hat face",
    "partying face",
    "disguised face",
    "smiling face with sunglasses",
    "nerd face",
    "face with monocle",
    "confused face",
    "face with diagonal mouth",
    "worried face",
    "slightly frowning face",
    "frowning face",
    "frowning face",
    "face with open mouth",
    "hushed face",
    "astonished face",
    "flushed face",
    "pleading face",
    "face holding back tears",
    "frowning face with open mouth",
    "anguished face",
    "fearful face",
    "anxious face with sweat",
    "sad but relieved face",
    "crying face",
    "loudly crying face",
    "face screaming in fear",
    "confounded face",
    "persevering face",
    "disappointed face",
    "downcast face with sweat",
    "weary face",
    "tired face",
    "yawning face",
    "face with steam from nose",
    "enraged face",
    "angry face",
    "smiling face with horns",
    "angry face with horns"
]

emoji_size = (20, 20)
emoji_images = []

def load_emoji_images():
    img = np.asarray(Image.open('D:/ITBA/4-anio/SIA/sia-tp5/input/emojis.png').convert("L"))
    emojis_per_row = img.shape[1] / emoji_size[1]
    for i in range(len(emoji_names)):
        y = int((i // emojis_per_row) * emoji_size[0])
        x = int((i % emojis_per_row) * emoji_size[1])
        emoji_matrix = img[y:(y + emoji_size[1]), x:(x + emoji_size[0])] / 255
        emoji_vector = emoji_matrix.flatten()
        emoji_images.append(emoji_vector)

load_emoji_images()