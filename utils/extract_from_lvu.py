import json
import pandas as pd
from config.defaults import cfg

def extract(file_path):
    df = pd.read_csv(file_path)
    label_dict = {}
    movie_name_list = []
    data = df['class_id class_name youtube_id imdb_id'].tolist()
    for line in data:
        line_split = line.split(' ')
        label = line_split[0]
        movie_name = line_split[2]
        label_dict[movie_name] = label
        movie_name_list.append(movie_name)
    return movie_name_list, label_dict


def extract_info():
    path = ''
    train_list, train_label = extract(path)
    path = ''
    test_list, test_label = extract(path)
    path = ''
    val_list, val_label = extract(path)
    test_list = test_list + val_list
    split = {'train': train_list, 'test': test_list}
    label = {}
    label.update(train_label)
    label.update(test_label)
    label.update(val_label)
    with open(cfg.DATA.LVU_SPLIT, 'w') as file:
        json.dump(split, file)
    with open(cfg.DATA.LVU_LABEL, 'w') as file:
        json.dump(label, file)


if __name__ == '__main__':
    extract_info()
