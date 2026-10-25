from ultralytics import YOLO

# Charge un petit modèle YOLO (il se télécharge tout seul la première fois)
model = YOLO("yolo11n.pt")

# Lance la détection sur une image décodée
results = model("data/decoded/crf28/img00001.png", device="cpu", conf=0.25)
r = results[0]

# Affiche ce qui a été trouvé
print("Nombre d'objets détectés :", len(r.boxes))
for box in r.boxes:
    nom = model.names[int(box.cls)]
    print(nom, "- confiance :", round(float(box.conf), 2))

# Sauvegarde l'image avec les boîtes dessinées
r.save("results/test_yolo.jpg")
print("Image sauvegardée dans results/test_yolo.jpg")