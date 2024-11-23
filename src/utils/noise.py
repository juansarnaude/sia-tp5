import numpy as np

def salt_and_pepper(array, noise_level=0.1):
    # noise_level es la proporción de píxeles afectados por el ruido (entre 0 y 1).
    # Si le pasamos por ejemplo 0.12 (12%) afectaremos 4 pixeles ya que el 12% de 35 es 4.2 y siempre redondea para abajo
    if not isinstance(array, np.ndarray):
        raise TypeError("El input debe ser un array de NumPy.")
    if array.size != 35:
        raise ValueError("El tamaño del array debe ser 35.")
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError("El array debe estar compuesto únicamente por ceros y unos.")
    if not (0 <= noise_level <= 1):
        raise ValueError("El nivel de ruido debe estar entre 0 y 1.")
    
    # Copia del array original para no modificar el original
    noisy_array = array.copy()
    
    # Total de elementos a modificar
    num_noisy_elements = int(noise_level * noisy_array.size)
    
    # Índices aleatorios para aplicar sal y pimienta
    indices = np.random.choice(noisy_array.size, num_noisy_elements, replace=False)
    
    for idx in indices:
        noisy_array[idx] = 1 - noisy_array[idx]  # Invierte el valor (0 -> 1 o 1 -> 0)
    
    return noisy_array