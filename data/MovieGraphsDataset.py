import json
import os

import torch
from torch.utils.data import Dataset
from config.defaults import cfg
from itertools import permutations


class MovieGraphsDataset(Dataset):
    def __init__(self, train):
        self.character_to_idx = {}  # character_name - idx
        self.idx_to_character = {}  # idx - character_name
        self.edge_to_idx = {}  # edge(node1_idx, node2_idx) - idx, here node1_idx < node2_idx
        self.idx_to_edge = {}  # idx - edge(node1_idx, node2_idx)

        # video-clip-names [a]
        data_file = open(cfg.DATA.NO_MOVIE_OVERLAP_SPLIT_FILE_PATH, 'r')
        data_json_object = json.load(data_file)
        if train:
            self.video_scene_name = data_json_object['train']
        else:
            self.video_scene_name = data_json_object['test']
        # exclude not exist videos
        temp_video_scene_name = []
        for name in self.video_scene_name:
            if name[:9] not in cfg.DATA.EXCLUDE_LIST:
                temp_video_scene_name.append(name)
        self.video_scene_name = temp_video_scene_name

        # face-features {movie:{scene:{character:tensor}}
        self.face_features = {}
        if os.path.exists(cfg.MODEL.FACE_FEATURE_SAVE_PATH):
            self.face_features = torch.load(cfg.MODEL.FACE_FEATURE_SAVE_PATH)
        face_feature_dir = cfg.FEATURE.FACE_FEATURE_DIR
        for path, dirnames, filenames in os.walk(face_feature_dir):
            if len(self.face_features) > 0:
                break
            for filename in filenames:
                face_feature_file_path = os.path.join(face_feature_dir, filename)
                face_feature_file = open(face_feature_file_path, 'r')
                face_fature_json_object = json.load(face_feature_file)
                for key, value in face_fature_json_object.items():
                    video_name = key[:9]
                    scene_id = int(key[10:13])
                    if video_name not in self.face_features:
                        self.face_features[video_name] = {}
                    if scene_id not in self.face_features[video_name]:
                        self.face_features[video_name][scene_id] = {}
                    for feature_record in value:
                        character_name = feature_record['name']
                        character_feature = feature_record['feature']
                        if character_name not in self.character_to_idx:
                            idx = len(self.character_to_idx)
                            self.character_to_idx[character_name] = idx
                            self.idx_to_character[idx] = character_name
                        else:
                            idx = self.character_to_idx[character_name]
                            self.face_features[video_name][scene_id][idx] = torch.Tensor(character_feature)
                print('finished %s' % video_name)
            torch.save(self.face_features, cfg.MODEL.FACE_FEATURE_SAVE_PATH)
            torch.save(self.character_to_idx, cfg.MODEL.CHARACTER_TO_IDX)
            torch.save(self.idx_to_character, cfg.MODEL.IDX_TO_CHARACTER)
        print('finished face_feature read')

        self.character_to_idx = torch.load(cfg.MODEL.CHARACTER_TO_IDX)
        self.idx_to_character = torch.load(cfg.MODEL.IDX_TO_CHARACTER)

        # bert-features {movie:{scene:{edge_key: tensor}}}, edge_key: (character_0_idx, character_1_idx) character_0_idx < character_1_idx
        self.bert_features = {}
        # if os.path.exists(cfg.MODEL.BERT_FEATURE_SAVE_PATH):
        #     self.bert_features = torch.load(cfg.MODEL.BERT_FEATURE_SAVE_PATH)
        bert_feature_dir = cfg.FEATURE.BERT_FEATURE_DIR
        for path, dirnames, filenames in os.walk(bert_feature_dir):
            if len(self.bert_features) > 0:
                break
            for filename in filenames:
                bert_feature_file_path = os.path.join(bert_feature_dir, filename)
                bert_feature_file = open(bert_feature_file_path, 'r')
                bert_feature_json_object = json.load(bert_feature_file)
                video_name = filename[:9]
                if video_name not in self.bert_features:
                    self.bert_features[video_name] = {}
                for key, value in bert_feature_json_object.items():
                    scene_id = int(key[6: 9])
                    if scene_id not in self.bert_features[video_name]:
                        self.bert_features[video_name][scene_id] = {}
                    for feature in value:
                        if feature['name1'] not in self.character_to_idx:
                            idx = len(self.character_to_idx)
                            self.character_to_idx[feature['name1']] = idx
                            self.idx_to_character[idx] = feature['name1']
                        if feature['name2'] not in self.character_to_idx:
                            idx = len(self.character_to_idx)
                            self.character_to_idx[feature['name2']] = idx
                            self.idx_to_character[idx] = feature['name2']
                        character_0 = self.character_to_idx[feature['name1']]
                        character_1 = self.character_to_idx[feature['name2']]
                        # try:
                        #     character_0 = self.character_to_idx[feature['name1']]
                        #     character_1 = self.character_to_idx[feature['name2']]
                        # except:
                        #     if video_name in cfg.DATA.EXIST_LIST:
                        #         print(video_name)
                        #         print(feature['name1'])
                        #         print(feature['name2'])
                        #         print('caught something weird')
                        #     continue
                        # if character_0 < character_1:
                        #     edge_key = (character_0, character_1)
                        # else:
                        #     edge_key = (character_1, character_0)
                        edge_key = (character_0, character_1)
                        self.bert_features[video_name][scene_id][edge_key] = torch.Tensor(feature['feature'])
            torch.save(self.bert_features, cfg.MODEL.BERT_FEATURE_SAVE_PATH)
        print('finished bert_feature read')

        # background-features {movie:{scene:tensor}}
        self.background_features = {}
        # if os.path.exists(cfg.MODEL.BACKGROUND_FEATURE_SAVE_PATH):
        #     self.background_features = torch.load(cfg.MODEL.BACKGROUND_FEATURE_SAVE_PATH)
        background_feature_path = cfg.FEATURE.BACKRGOUND_FEATURE_Path
        background_feature_file = open(background_feature_path, 'r')
        background_feature_json_object = json.load(background_feature_file)
        if len(self.background_features) == 0:
            for key, value in background_feature_json_object.items():
                video_name = key[:9]
                scene_id = int(key[-3:])
                if video_name not in self.background_features:
                    self.background_features[video_name] = {}
                self.background_features[video_name][scene_id] = torch.Tensor(value)
            torch.save(self.background_features, cfg.MODEL.BACKGROUND_FEATURE_SAVE_PATH)
        print('finished background_features read')

        # label {movie:{scene:{character_pair:label}}, character_pair:(character_1, character_2) character_1 < character_2
        self.label = {}
        label_dir = cfg.DATA.LABEL_DIR
        now_scene = ''
        for path, dirnames, filenames in os.walk(label_dir):
            for filename in filenames:
                now_movie = filename[:9]
                if now_movie not in self.label:
                    self.label[now_movie] = {}
                label_file = open(os.path.join(label_dir, filename), 'r')
                for line in label_file.readlines():
                    if line == '\n':
                        continue
                    elif line[:5] == 'scene':
                        now_scene = int(line[6:9])
                        if now_scene not in self.label[now_movie]:
                            self.label[now_movie][now_scene] = {}
                    else:
                        if line.split(';')[0] not in self.character_to_idx:
                            idx = len(self.character_to_idx)
                            self.character_to_idx[line.split(';')[0]] = idx
                            self.idx_to_character[idx] = line.split(';')[0]
                        if line.split(';')[1] not in self.character_to_idx:
                            idx = len(self.character_to_idx)
                            self.character_to_idx[line.split(';')[1]] = idx
                            self.idx_to_character[idx] = line.split(';')[1]
                        character_0 = self.character_to_idx[line.split(';')[0]]
                        character_1 = self.character_to_idx[line.split(';')[1]]
                        label = int(line.split(';')[2]) + 1  # begin from 1
                        # ["Leader-Sub.", "Colleague", "Service", "Parent-offs", "Sibling","Couple", "Friend", "Opponent", "Stranger"]
                        # if character_0 < character_1:
                        #     self.label[now_movie][now_scene][(character_0, character_1)] = label
                        # else:
                        #     self.label[now_movie][now_scene][(character_1, character_0)] = label
                        self.label[now_movie][now_scene][(character_0, character_1)] = label
        print('finished label read')

    def __len__(self):
        return len(self.video_scene_name)

    def __getitem__(self, idx):
        end = False
        video_scene_name = self.video_scene_name[idx]
        video_name = video_scene_name[:9]
        scene_id = int(video_scene_name[16:19])
        if idx == len(self.video_scene_name) - 1:
            end = True
        elif self.video_scene_name[idx + 1][:9] != video_name:
            end = True

        bert_features = self.bert_features[video_name][scene_id]

        background_feature = self.background_features[video_name][scene_id]

        face_features = []
        face_features = self.face_features[video_name][scene_id]

        graph_features_face_dim, graph_feature_edge_idx, adj = self.construct_graph_feature(face_features)

        label = []  # if no relation annotation in label_file, then assign 0
        for edge_idx in graph_feature_edge_idx:
            edge_key = self.idx_to_edge[edge_idx]
            if edge_key not in self.label[video_name][scene_id]:
                label.append(torch.tensor(0, dtype=torch.long))
            else:
                label.append(torch.tensor(self.label[video_name][scene_id][edge_key], dtype=torch.long))

        bert_features = self.construct_bert_feature(bert_features, graph_feature_edge_idx)
        graph_feature_edge_idx = torch.tensor(graph_feature_edge_idx, dtype=torch.long)
        label = torch.stack(label)
        label = label.reshape(label.shape[0], 1)

        return bert_features, background_feature, graph_features_face_dim, graph_feature_edge_idx, adj, label, end

    # select related bert_feature from overall bert_features
    def construct_bert_feature(self, bert_features, graph_feature_edge_idx):
        bert_feature = []
        for i in graph_feature_edge_idx:
            if self.idx_to_edge[i] not in bert_features:
                bert_feature.append(torch.zeros(size=(1, cfg.FEATURE.TEXT_DIM)))
            else:
                temp_bert = []
                temp_bert.append(torch.mean(bert_features[self.idx_to_edge[i]], dim=0))
                temp_bert = torch.stack(temp_bert)
                bert_feature.append(temp_bert)
        return torch.stack(bert_feature)

    def construct_graph_feature(self, face_features):
        #  construct the graph of edges
        character_idxs = []
        graph_features = []
        graph_feature_edge_idx = []

        # get the permutation of nodes, representing edges
        # exists duplication, for recording the neighboring edges based on the mutual node
        for key in face_features.keys():
            character_idxs.append(key)
        character_permutations = list(permutations(character_idxs, 2))
        size = len(character_permutations)

        # construct adjacent matrix based on the permutation of nodes
        adj = torch.zeros(size=(cfg.MODEL.MAX_EDGES, cfg.MODEL.MAX_EDGES))
        edge_list_same_node_0 = []
        edge_list_all = []
        now_node_0 = -1
        for permutation in character_permutations:
            node_0 = permutation[0]
            node_1 = permutation[1]
            # record the edge
            if node_0 < node_1:
                edge_key = (node_0, node_1)
            else:
                edge_key = (node_1, node_0)
            if edge_key in self.edge_to_idx:
                edge_idx = self.edge_to_idx[edge_key]
            else:
                edge_idx = len(self.edge_to_idx)
                self.edge_to_idx[edge_key] = edge_idx
                self.idx_to_edge[edge_idx] = edge_key
            # construct edge_feature without edge duplication
            if edge_idx not in edge_list_all:
                edge_list_all.append(edge_idx)
                character_feature_fusion = torch.add(face_features[node_0], face_features[node_1])
                graph_features.append(character_feature_fusion)
                graph_feature_edge_idx.append(edge_idx)
            # determine whether to construct adj
            if node_0 != now_node_0 and now_node_0 != -1:
                # a new starting node begins, now deal with the previous neighboring edges
                # remove duplicate in edge_list
                for edge_permutation in permutations(edge_list_same_node_0, 2):
                    adj[edge_permutation[0]][edge_permutation[1]] = 1
                edge_list_same_node_0.clear()
            edge_list_same_node_0.append(edge_idx)
            now_node_0 = node_0

        # deal with the last edge_list
        for edge_permutation in permutations(edge_list_same_node_0, 2):
            adj[edge_permutation[0]][edge_permutation[1]] = 1
            adj[edge_permutation[1]][edge_permutation[0]] = 1

        return torch.stack(graph_features), graph_feature_edge_idx, adj
