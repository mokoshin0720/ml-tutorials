from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# データの作成
input_date = []
output_date = []
file_path = 'date.txt'
with open(file_path, "r") as f:
    date_list = f.readlines()
    for date in date_list:
        date = date[:-1]
        input_date.append(date.split("_")[0])
        output_date.append("_" + date.split("_")[1])

# char2idの作成
char2id = {}
for input_chars, output_chars in zip(input_date, output_date):
    for c in input_chars:
        if not c in char2id:
            char2id[c] = len(char2id)
    for c in output_chars:
        if not c in char2id:
            char2id[c] = len(char2id)

def create_data(input_date, output_date, char2id):
    input_data = []
    output_data = []

    for input_chars, output_chars in zip(input_date, output_date):
        input_data.append([char2id[c] for c in input_chars])
        output_data.append([char2id[c] for c in output_chars])

    train_x, test_x, train_y, test_y = train_test_split(input_data, output_data, train_size=0.7)
    return train_x, test_x, train_y, test_y

def train2batch(input_data, output_data, batch_size=100):
    input_batch = []
    output_batch = []
    input_shuffle, output_shuffle = shuffle(input_data, output_data)
    for i in range(0, len(input_data), batch_size):
      input_batch.append(input_shuffle[i:i+batch_size])
      output_batch.append(output_shuffle[i:i+batch_size])
    return input_batch, output_batch