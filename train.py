from ultralytics.models import RTDETR

if __name__ == "__main__":
    model = RTDETR(model="rtdetr-l-NWD.yaml")
    model.train(pretrained=True, data="data.yaml", epochs=300, batch=4, device=0, imgsz=640, workers=0)
