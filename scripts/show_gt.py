import xml.etree.ElementTree as ET
import cv2

root = ET.parse("data/MVI_20011/MVI_20011.xml").getroot()
img = cv2.imread("data/decoded/crf28/img00001.png")

# Zones ignorées par le dataset : dessinées en rouge
nb_zones = 0
for region in root.iter("ignored_region"):
    for b in region.iter("box"):
        x, y, w, h = [float(b.attrib[k]) for k in ("left", "top", "width", "height")]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        nb_zones += 1

# Véhicules annotés dans l'image numéro 1 : dessinés en vert
nb_vehicules = 0
types = {}
for frame in root.iter("frame"):
    if int(frame.attrib["num"]) == 1:
        for target in frame.iter("target"):
            b = target.find("box")
            x, y, w, h = [float(b.attrib[k]) for k in ("left", "top", "width", "height")]
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            nb_vehicules += 1
            attr = target.find("attribute")
            t = attr.attrib.get("vehicle_type", "?") if attr is not None else "?"
            types[t] = types.get(t, 0) + 1

print("Zones ignorées :", nb_zones)
print("Véhicules annotés dans l'image 1 :", nb_vehicules)
print("Types :", types)
cv2.imwrite("results/gt_frame1.jpg", img)
print("Image sauvegardée dans results/gt_frame1.jpg")