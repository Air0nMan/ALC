
import torch
from torchvision.models import efficientnet_b3
from torchvision.models import EfficientNet_B3_Weights
from torchvision.io import decode_image, read_image
import matplotlib.pyplot as plt
import requests
import json
# Funcion que permite graficar las imagenes y sus embeddings asociados
def plot_images_and_embeddings(images, embeddings, filenames):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for idx in range(2):
        # Plot original image
        axs[idx, 0].imshow(images[idx].numpy().transpose(1, 2, 0))
        axs[idx, 0].set_title(f"Imagen original: {filenames[idx]}")
        axs[idx, 0].axis('off')

        # Plot embedding
        embedding_image = embeddings[idx].reshape((48, 32))  # cambia el shape para visualizarlo
        axs[idx, 1].imshow(embedding_image, cmap='viridis')
        axs[idx, 1].set_title(f"Embedding {filenames[idx]}")
        axs[idx, 1].axis('off')
    plt.tight_layout()
    plt.show()


# Leer las etiquetas del dataset ImageNet
# URL de origen
labels_url = ("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json")
# Levantar el contenido de la url
response = requests.get(labels_url)
response.raise_for_status()  

# cargar el contenido como JSON
imagenet_labels = json.loads(response.text)

weights = EfficientNet_B3_Weights.DEFAULT
preprocess = weights.transforms()
model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
model.eval()
imagenes = ['cat.7.jpg', 'dog.4.jpg']

# Definir listas para reservar las imagenes, embedding y luego graficar
images_processed = []
embeddings = []

for imagen in imagenes:
    i = read_image(imagen)
    with torch.no_grad():
        # Aplicar la transformacion para preparar la imagen al formato que entiende el modelo
        processed_image = preprocess(i).unsqueeze(0)
        images_processed.append(i)

        i = preprocess(i).unsqueeze(0)
        # Obtener la salida del modelo con las probabilidades de cada clase de IMAGENET
        outputs = model(processed_image)
        # Identificar la clase con la maxima probabilidad
        _, predicted_idx = torch.max(outputs, 1)
        # Mostrar la etiqueta de la maxima probabilidad
        predicted_label = imagenet_labels[predicted_idx.item()]
        print(f"Clase predicha por el modelo para {imagen}: {predicted_label}")

        # Extraer los embeddings
        features = model.features(processed_image) # model.features sería la salida del Body
        pooled_features = model.avgpool(features) # paso anterior al fully connected que comprime el tamaño
        flattened = torch.flatten(pooled_features, 1).detach().cpu().numpy()
        embeddings.append(flattened[0])

# Plotear la imagen y los embeddings
plot_images_and_embeddings(images_processed, embeddings, imagenes)


