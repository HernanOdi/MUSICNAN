from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np

def upload_image(request):
    if request.method == 'POST' and request.FILES['imagen']:
        # Guardar la imagen cargada
        uploaded_image = request.FILES['imagen']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        uploaded_image_url = fs.url(filename)

        # Procesar la imagen con menor suavización del filtro bilateral
        image_path = fs.path(filename)
        preprocessed_image = preprocess_image(image_path)

        # Guardar la imagen procesada para revisarla
        processed_image_path = fs.path('processed_' + uploaded_image.name)
        cv2.imwrite(processed_image_path, preprocessed_image)

        return render(request, 'converter/home.html', {
            'uploaded_image_url': uploaded_image_url,
            'processed_image_url': fs.url('processed_' + uploaded_image.name)
        })

    return render(request, 'converter/home.html')

def preprocess_image(image_path):
    # Cargar la imagen en color
    image = cv2.imread(image_path)

    # Paso 1: Redimensionar la imagen con preservación de aspecto
    target_size = (256, 256)
    resized_image = resize_with_aspect_ratio(image, target_size)

    # Paso 2: Eliminación de ruido usando filtro bilateral con menor suavización
    # Reducimos sigmaColor y sigmaSpace para mantener más detalles
    denoised_image = cv2.bilateralFilter(resized_image, d=9, sigmaColor=50, sigmaSpace=50)

    # Devolvemos la imagen en color con menor suavización
    return denoised_image

def resize_with_aspect_ratio(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))

    # Agregar bordes para mantener el tamaño objetivo sin distorsionar
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image
