import numpy as np
from PIL import Image

NUMBER_OF_EMOJIS = 8

emoji_size = (20, 20)
emoji_images = []

def load_emoji_images():
    img = np.asarray(Image.open('./input/emojis.png').convert("L"))
    emojis_per_row = img.shape[1] / emoji_size[1]
    for i in range(NUMBER_OF_EMOJIS):
        y = int((i // emojis_per_row) * emoji_size[0])
        x = int((i % emojis_per_row) * emoji_size[1])
        emoji_matrix = img[y:(y + emoji_size[1]), x:(x + emoji_size[0])] / 255
        emoji_vector = emoji_matrix.flatten()
        emoji_images.append(emoji_vector)

def vector_to_emoji(vector, emoji_size=(20, 20), show=True):
    """
    Convierte un vector de emoji en una imagen y opcionalmente la muestra.

    Parámetros:
    vector: numpy array 1D normalizado (0-1)
    emoji_size: tupla (height, width), por defecto (20, 20)
    show: boolean, si True muestra la imagen

    Retorna:
    PIL Image: imagen del emoji reconstruida
    """
    # Verificar dimensiones

    expected_size = emoji_size[0] * emoji_size[1]
    if len(vector) != expected_size:
        raise ValueError(f"El vector debe tener {expected_size} elementos")

    # Reconstruir matriz
    matrix = vector.reshape(emoji_size)

    # Convertir a imagen
    image_array = (matrix * 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode='L')

    if show:
        image.show()

    return image

load_emoji_images()