
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

inputs = tokenizer("使用语言模型来预测下一个词", return_tensors="pt")
outputs = model(**inputs)
print(outputs)