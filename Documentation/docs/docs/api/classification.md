# 🖼️ CIFAR-10 Image Classifier Service

This service provides a basic image classification API.

## Endpoint

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/image/classify` | Classifies an uploaded image into one of 10 categories. |

## Functionality

* **Core Model:** Custom-trained **CNN**
* **Dataset:** CIFAR-10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
* **Input:** 32x32 color image
* **Output:** JSON with `predicted_label` and `confidence`

## Use Case
Basic content filtering, academic tasks, or a starting point for visual pipelines.
