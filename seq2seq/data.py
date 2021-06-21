import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# 数字の文字をID化
char2id = {str(i) : i for i in range(10)}
char2id.update({" ":10, "-":11, "_":12})

id2char = {str(i): str(i) for i in range(10)}
id2char.update({"10":"", "11":"-", "12":""})

# 空白込みの3桁の数字をランダムに生成
def generate_number():
    number = [random.choice(list("0123456789")) for _ in range(random.randint(1, 3))]
    return int("".join(number))

# 系列の長さを揃えるために空白でpadding
def add_padding(number, is_input=True):
    if is_input:
        number = "{: <7}".format(number)
    else:
        number = "{: <5s}".format(number)
    return number

def create_data():
    input_data = []
    output_data = []
    while len(input_data) < 50000:
        x = generate_number()
        y = generate_number()
        z = x-y
        input_char = add_padding(str(x) + "-" + str(y))
        output_char = add_padding("_" + str(z), is_input=False)
        # データをIDに変換
        input_data.append([char2id[c] for c in input_char])
        output_data.append([char2id[c] for c in output_char])
    # データを7:3に分ける
    train_x, test_x, train_y, test_y = train_test_split(input_data, output_data)
    return train_x, test_x, train_y, test_y

def train2batch(input_data, output_data, batch_size=100):
    input_batch = []
    output_batch = []
    input_shuffle, output_shuffle = shuffle(input_data, output_data)
    for i in range(0, len(input_data), batch_size):
        input_batch.append(input_shuffle[i:i+batch_size])
        output_batch.append(output_shuffle[i:i+batch_size])
    return input_batch, output_batch