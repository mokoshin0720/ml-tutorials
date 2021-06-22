from data import create_data, train2batch, input_date, output_date, char2id
from models import Encoder, AttentionDecoder
import torch
import torch.nn as nn
import torch.optim as optim

def train():
    all_losses = []
    print("training...")
    for epoch in range(1, EPOCH_NUM+1):
        epoch_loss = 0
        input_batch, output_batch = train2batch(train_x, train_y, batch_size=BATCH_NUM)

        for i in range(len(input_batch)):
            # 勾配の初期化
            encoder_optimizer.zero_grad()
            attn_decoder_optimizer.zero_grad()
            # データをtensorに変換
            input_tensor = torch.tensor(input_batch[i])
            output_tensor = torch.tensor(output_batch[i])
            # Encoderの順伝搬
            hs, h = encoder(input_tensor)
            # AttentionDecoderのインプット&正解データ
            source = output_tensor[:, :-1]
            target = output_tensor[:, 1:]

            loss = 0
            decoder_output, _, attention_weight = attn_decoder(source, hs, h)

            for j in range(decoder_output.size()[1]):
                loss += criterion(decoder_output[:, j, :], target[:, j])
            
            epoch_loss += loss.item()
            loss.backward()
            encoder_optimizer.step()
            attn_decoder_optimizer.step()

        print("Epoch %d: %.2f" % (epoch, epoch_loss))
        all_losses.append(epoch_loss)
        if epoch_loss < 0.1: break
    
    torch.save(encoder.state_dict(), 'models/encoder.pth')
    torch.save(attn_decoder.state_dict(), 'models/decoder.pth')
    print("Done")

if __name__ == '__main__':
    train_x, test_x, train_y, test_y = create_data(input_date, output_date, char2id)

    # 諸々のパラメータなど
    embedding_dim = 200
    hidden_dim = 128
    vocab_size = len(char2id)
    BATCH_NUM = 100 
    EPOCH_NUM = 10

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    attn_decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, BATCH_NUM)

    criterion = nn.CrossEntropyLoss()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    attn_decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=0.001)
    train()