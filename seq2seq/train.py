import torch
import torch.nn as nn
import torch.optim as optim
from models import Encoder, Decoder, vocab_size, embedding_dim, hidden_dim
from data import create_data, train2batch

BATCH_NUM = 100
EPOCH_NUM = 100

def train():
    all_losses = []
    print("training...")
    for epoch in range(1, EPOCH_NUM+1):
        epoch_loss = 0
        input_batch, output_batch = train2batch(train_x, train_y, batch_size=BATCH_NUM)
        # batchごとに処理を行う
        for i in range(len(input_batch)):
            # 勾配の初期化
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # データをtensorに変換
            input_tensor = torch.tensor(input_batch[i])
            output_tensor = torch.tensor(output_batch[i])
            # Encoderの順伝搬
            encoder_state = encoder(input_tensor)
            # Decoderにinputするデータ
            source = output_tensor[:,:-1]
            # Decoderの教師データ
            target = output_tensor[:, 1:]
        
            loss = 0
            decoder_output, _ = decoder(source, encoder_state)

            for j in range(decoder_output.size()[1]):
                loss += criterion(decoder_output[:, j, :], target[:, j])

            epoch_loss += loss.item()
            # 誤差逆伝搬
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

        print("Epoch %d: %.2f" % (epoch, epoch_loss))
        all_losses.append(epoch_loss)
        if epoch_loss < 1: break

    torch.save(encoder.state_dict(), 'seq2seq/models/encoder.pth')
    torch.save(decoder.state_dict(), 'seq2seq/models/decoder.pth')

    print("Done")

if __name__ == '__main__':
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim)

    criterion = nn.CrossEntropyLoss()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    train_x, test_x, train_y, test_y = create_data()

    train()