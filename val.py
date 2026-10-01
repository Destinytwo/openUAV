from ultralytics.models import RTDETR

if __name__ == "__main__":
    model = RTDETR(
        model="C:\\Funny\\ultralytics-main\\ultralytics-main\\RT-DETR\\Origin-PKI-P-NWD0.4\\weights\\best.pt"
    )
    model.val(data="data.yaml", batch=16, device="0", imgsz=640, workers=4)
