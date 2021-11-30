from torchmetrics import ROUGEScore, WER

targets = "is your name John".split()
preds = "My name is John".split()

rouge = ROUGEScore()
print(rouge(preds, targets))

predictions = ["this is the prediction", "there is an other sample"]
references = ["this is the reference", "there is another one"]

metric = WER()
print(metric(predictions, references))