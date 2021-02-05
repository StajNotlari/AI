import inception

inception.download()

model = inception.Inception()

def classify(image_path):
    pred = model.classify(image_path=image_path)
    model.print_scores(pred=pred, k=10, only_first_name=True)

classify(image_path='D:/Isler/Yapay Zeka/GitHub/Ornekler/7-imagenetInceptionNesneTanima/images/3.jpg')


