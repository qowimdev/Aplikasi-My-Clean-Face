

url = "https://img.okezone.com/content/2020/11/06/611/2305199/kesal-dengan-komedo-di-hidung-ini-5-cara-rumahan-untuk-menghilangkannya-Pvz6lc00bB.jpg"

import torch
import torchvision

print(torchvision.__version__)
print(torch.__version__)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt', force_reload=True)

results = model("levle0_4.jpg")
results.save()