from  ultralytics import YOLO
from PIL import Image
model = YOLO('best.pt')
result = model("Screenshot 2024-09-18 173549.png")
for r in result:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('test.png')