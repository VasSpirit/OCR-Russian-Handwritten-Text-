import cv2
import numpy as np
from src.contact_classifier import ContactClassifier, ContactClass


def test_contact_classifier_plus():
    img = np.full((80, 80), 255, dtype=np.uint8)
    cv2.line(img, (20, 40), (60, 40), 0, 4)
    cv2.line(img, (40, 20), (40, 60), 0, 4)
    result = ContactClassifier().classify(img)
    assert result.text == ContactClass.PLUS.value
