import json

from config.defaults import cfg
import numpy as np


def groupby(video_scene_name, movie_list):
    over_all_dict = {}
    over_all_list = []
    for item in video_scene_name:
        video_name = item[:9]
        if video_name not in movie_list:
            continue
        if video_name not in over_all_dict:
            over_all_dict[video_name] = []
        over_all_dict[video_name].append(item)
    # sort by scene
    for movie in over_all_dict.keys():
        over_all_list.extend(sorted(over_all_dict[movie], key=lambda x: int(x[16:19])))
    return over_all_list


def rearrange_split():
    data_file = open(cfg.DATA.SPLIT_FILE_PATH, 'r')
    data_json_object = json.load(data_file)
    video_scene_name_train = data_json_object['train']
    video_scene_name_test = data_json_object['test']
    all_video_scene_list = video_scene_name_train + video_scene_name_test
    movie_data_file = open(cfg.DATA.MOVIE_SPLIT_FILE_PATH, 'r')
    data_json_object = json.load(movie_data_file)
    data_json_object_train = data_json_object['train']
    data_json_object_test = data_json_object['test']
    data_json_object_val = data_json_object['val']
    data_json_object_test = data_json_object_test + data_json_object_val
    new_list_train = groupby(all_video_scene_list, data_json_object_train)
    new_list_test = groupby(all_video_scene_list, data_json_object_test)
    new_split = {'train': new_list_train, 'test': new_list_test}
    with open(cfg.DATA.NO_MOVIE_OVERLAP_SPLIT_FILE_PATH, 'w') as file:
        json.dump(new_split, file)


if __name__ == '__main__':
    rearrange_split()
