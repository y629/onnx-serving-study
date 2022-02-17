import os, sys
import numpy as np
import grpc
import joblib
import numpy as np
from PIL import Image
from pydantic import BaseModel
from proto import onnx_ml_pb2, predict_pb2, prediction_service_pb2_grpc
from torchvision import transforms
import torch

DEFAULT_IMAGE_PATH = './data/dog.jpg'

# 画像オープン用の関数
# 指定ファイルに画像がなければデフォルトで設定した画像を開いて返す
def open_image(img_path):
  try:
    img = Image.open(img_path)
    print(f"Loaded: {img_path}")
  except(FileNotFoundError): # 指定したファイルがなかった例外
    print(f"Not Found: {img_path}\nLoad default image: {DEFAULT_IMAGE_PATH}...")
    img = Image.open(DEFAULT_IMAGE_PATH)
  return img

def main():
  # 推論器に入れる前の前処理
  preprocess = transforms.Compose([
      transforms.Resize(299),
      transforms.CenterCrop(299),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])


  # 画像データへのパスを取得
  try:
    img_path = sys.argv[1]
  except(IndexError): # 可変長引数を与えなかった例外
    img_path = './data/dog.jpg'
  
  # 画像データをオープン
  img_raw = open_image(img_path)

  # 前処理の実行
  preprocessed = preprocess(img_raw)
  preprocessed = torch.unsqueeze(preprocessed, 0).numpy() # add batch dim

  # ONNX形式のモデルに入力できる形にする
  input_tensor = onnx_ml_pb2.TensorProto()
  input_tensor.dims.extend(preprocessed.shape)
  input_tensor.data_type = 1
  input_tensor.raw_data = preprocessed.tobytes()

  # 推論器へのリクエストメッセージ作成
  request_message = predict_pb2.PredictRequest()
  request_message.inputs["actual_input_1"].data_type = input_tensor.data_type
  request_message.inputs["actual_input_1"].dims.extend(preprocessed.shape)
  request_message.inputs["actual_input_1"].raw_data = input_tensor.raw_data

  # 通信関連の準備
  # Dockerで実行時はアドレスとポートは環境変数から取得，デフォルト値はローカル実行時の値を指定
  address = os.getenv("API_ADDRESS", "localhost")
  port = int(os.getenv("GRPC_PORT", 50051))
  serving_address = f'{address}:{port}'
  channel = grpc.insecure_channel(serving_address)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  # リクエストを送ってレスポンスを得る
  # 推論器サーバが起動していないとエラーになる
  print("request sending...")
  response = stub.Predict(request_message)
  print("response received...")

  # 推論結果から出力を取得
  # まず予測値を取り出して，numpy配列にする
  output = np.frombuffer(response.outputs["output1"].raw_data, dtype=np.float32)

  # 予測値を確率に変換するためにsoftmaxする
  probabilities = torch.nn.functional.softmax(torch.tensor(output), dim=0)

  # 予測確率から予測ラベルを出力
  # Read the categories
  with open("./data/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
  # Show top categories per image
  print('Top5 label prob')
  top5_prob, top5_catid = torch.topk(probabilities, 5)
  for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

if __name__=="__main__":
  main()