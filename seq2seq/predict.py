import torch
import pandas as pd
from data import train2batch
from train import BATCH_NUM
from data import create_data, char2id, id2char
from models import Encoder, Decoder, vocab_size, embedding_dim, hidden_dim

# Decoderのアウトプットのtensorから要素が最大のインデックスを返す。
def get_max_index(decoder_output):
    results = []
    for h in decoder_output:
        results.append(torch.argmax(h))
    return torch.tensor(results).view(BATCH_NUM, 1)

def predict():
    predicts = []
    for i in range(len(test_input_batch)):
        with torch.no_grad(): # 勾配計算をさせない
            # Enc-Decに入力文字を渡す
            encoder_state = encoder(input_tensor[i])
            start_char_batch = [[char2id["_"]] for _ in range(BATCH_NUM)]
            decoder_input_tensor = torch.tensor(start_char_batch)

            decoder_hidden = encoder_state
            # batch毎の結果を結合するための入れ物を定義
            batch_tmp = torch.zeros(100, 1, dtype=torch.long)

            for _ in range(5):
                decoder_output, decoder_hidden = decoder(decoder_input_tensor, decoder_hidden)
                # 予測文字を取得しつつ、そのまま次のdecoderのinputとなる
                decoder_input_tensor = get_max_index(decoder_output.squeeze())
                # batch毎の結果を予測順に結合
                batch_tmp = torch.cat([batch_tmp, decoder_input_tensor], dim=1)
        
            predicts.append(batch_tmp[:, 1:])
    return predicts

def cal_accuracy(predicts):
    row = []
    for i in range(len(test_input_batch)):
        batch_input = test_input_batch[i]
        batch_output = test_output_batch[i]
        batch_predict = predicts[i]

        for inp, output, predict in zip(batch_input, batch_output, batch_predict):
            x = [id2char[str(idx)] for idx in inp]
            y = [id2char[str(idx)] for idx in output]
            p = [id2char[str(idx.item())] for idx in predict]

            x_str = "".join(x)
            y_str = "".join(y)
            p_str = "".join(p)

            judge = "O" if y_str == p_str else "X"
            row.append([x_str, y_str, p_str, judge])
    predict_df = pd.DataFrame(row, columns=["input", "answer", "predict", "judge"])
    
    print(len(predict_df.query('judge=="O"')) / len(predict_df))
    print(predict_df.query('judge == "X"').head(10))

if __name__ == '__main__':
    # 学習済みモデルの読み込み
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
    encoder_path = 'seq2seq/models/encoder.pth'
    decoder_path = 'seq2seq/models/decoder.pth'
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    encoder.eval()
    decoder.eval()

    # 評価用データ
    _, test_x, _, test_y = create_data()
    test_input_batch, test_output_batch = train2batch(test_x, test_y)
    input_tensor = torch.tensor(test_input_batch)

    predicts = predict()
    cal_accuracy(predicts)