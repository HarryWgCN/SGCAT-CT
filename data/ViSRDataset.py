import json
import os

import torch
from torch.utils.data import Dataset
from config.defaults import cfg
from itertools import permutations
import numpy as np

class ViSRDataset(Dataset):
    def __init__(self, train):
        self.character_to_idx = {}  # {movie:{character_name:idx}}
        self.idx_to_character = {}  # {movie:{idx:character_name}}
        self.edge_to_idx = {}  # movie: edge(node1_idx, node2_idx) - idx, here node1_idx < node2_idx
        self.idx_to_edge = {}  # movie: idx - edge(node1_idx, node2_idx)

        # video_scene_names [movie_scene]
        data_file = open(cfg.DATA.VISR_SPLIT_PATH, 'r')
        data_json_object = json.load(data_file)
        if train:
            self.video_scene_name = data_json_object['train']
        else:
            self.video_scene_name = data_json_object['test']

        # add scene
        temp_video_scene_name = {}
        temp_video_scene_list = []
        if os.path.exists(cfg.MODEL.VISR_VIDEO_SCENE_NAME_TRAIN_PATH) and train:
            temp_video_scene_list = torch.load(cfg.MODEL.VISR_VIDEO_SCENE_NAME_TRAIN_PATH)
        if os.path.exists(cfg.MODEL.VISR_VIDEO_SCENE_NAME_TEST_PATH) and not train:
            temp_video_scene_list = torch.load(cfg.MODEL.VISR_VIDEO_SCENE_NAME_TEST_PATH)
        for record in self.video_scene_name:
            if len(temp_video_scene_list) > 0 and len(temp_video_scene_name) == 0:
                break
            video_dir = os.path.join(cfg.DATA.VISR_FRAME_DIR, record)
            for path, dirnames, filenames in os.walk(video_dir):
                for filename in filenames:
                    if record not in temp_video_scene_name:
                        temp_video_scene_name[record] = []
                    temp_video_scene_name[record].append(f'{record}_{filename.split(".")[0].split("@")[-1]}')
        for key, value in temp_video_scene_name.items():
            if len(temp_video_scene_list) > 0 and len(temp_video_scene_name) == 0:
                break
            value.sort(key=lambda x: int(x.split('_')[-1]))
            temp_video_scene_list += value
        self.video_scene_name = temp_video_scene_list
        if train:
            torch.save(self.video_scene_name, cfg.MODEL.VISR_VIDEO_SCENE_NAME_TRAIN_PATH)
        if not train:
            torch.save(self.video_scene_name, cfg.MODEL.VISR_VIDEO_SCENE_NAME_TEST_PATH)

        # face-features {movie:{scene:{character:tensor}}
        self.face_features = {}
        if os.path.exists(cfg.MODEL.VISR_FACE_FEATURE_SAVE_PATH):
            self.face_features = torch.load(cfg.MODEL.VISR_FACE_FEATURE_SAVE_PATH)
        face_feature_dir = cfg.FEATURE.VISR_FACE_FEATURE_DIR
        for path, dirnames, filenames in os.walk(face_feature_dir):
            if len(self.face_features) > 0:
                break
            for dirname in dirnames:
                movie_face_feature_dir = os.path.join(face_feature_dir, dirname)
                for path_, dirnames_, filenames_ in os.walk(movie_face_feature_dir):
                    for filename_ in filenames_:
                        face_feature_file_path = os.path.join(movie_face_feature_dir, filename_)
                        face_feature_object = np.load(face_feature_file_path)
                        video_name = int(dirname)
                        scene_id = int(filename_.split('@')[2].split('_')[0])
                        person_name = int(filename_.split('@')[2].split('_')[2].split('.')[0])
                        if video_name not in self.face_features:
                            self.face_features[video_name] = {}
                        if scene_id not in self.face_features[video_name]:
                            self.face_features[video_name][scene_id] = {}
                        if video_name not in self.character_to_idx:
                            self.character_to_idx[video_name] = {}
                        if video_name not in self.idx_to_character:
                            self.idx_to_character[video_name] = {}

                        if person_name not in self.character_to_idx[video_name]:
                            idx = len(self.character_to_idx[video_name])
                            self.character_to_idx[video_name][person_name] = idx
                            self.idx_to_character[video_name][idx] = person_name
                        else:
                            idx = self.character_to_idx[video_name][person_name]
                        self.face_features[video_name][scene_id][idx] = torch.Tensor(face_feature_object)
                    break
                print('finished %s' % video_name)
            torch.save(self.face_features, cfg.MODEL.VISR_FACE_FEATURE_SAVE_PATH)
            torch.save(self.character_to_idx, cfg.MODEL.VISR_CHARACTER_TO_IDX)
            torch.save(self.idx_to_character, cfg.MODEL.VISR_IDX_TO_CHARACTER)
            break
        print('finished face_feature read')

        self.character_to_idx = torch.load(cfg.MODEL.VISR_CHARACTER_TO_IDX)
        self.idx_to_character = torch.load(cfg.MODEL.VISR_IDX_TO_CHARACTER)

        # bert-features {movie: tensor}
        self.bert_features = {}
        if os.path.exists(cfg.MODEL.VISR_BERT_FEATURE_SAVE_PATH):
            self.bert_features = torch.load(cfg.MODEL.VISR_BERT_FEATURE_SAVE_PATH)
        bert_feature_dir = cfg.FEATURE.VISR_BERT_FEATURE_DIR
        for path, dirnames, filenames in os.walk(bert_feature_dir):
            if len(self.bert_features) > 0:
                break
            for filename in filenames:
                bert_feature_file_path = os.path.join(bert_feature_dir, filename)
                bert_feature_json_object = np.loadtxt(bert_feature_file_path)
                video_name = filename.split('.')[0]
                self.bert_features[video_name] = torch.Tensor(bert_feature_json_object)
            torch.save(self.bert_features, cfg.MODEL.VISR_BERT_FEATURE_SAVE_PATH)
        print('finished bert_feature read')

        # background-features {movie:{scene:tensor}}
        self.background_features = {}
        if os.path.exists(cfg.MODEL.VISR_BACKGROUND_FEATURE_SAVE_PATH):
            self.background_features = torch.load(cfg.MODEL.VISR_BACKGROUND_FEATURE_SAVE_PATH)
        background_feature_dir = cfg.FEATURE.VISR_BACKRGOUND_FEATURE_DIR
        for path, dirnames, filenames in os.walk(background_feature_dir):
            if len(self.background_features) > 0:
                break
            for filename in filenames:
                background_feature_file_path = os.path.join(background_feature_dir, filename)
                background_feature_json_object = np.loadtxt(background_feature_file_path)
                video_name = filename.split('.')[0]
                self.background_features[video_name] = torch.Tensor(background_feature_json_object)
            torch.save(self.background_features, cfg.MODEL.VISR_BACKGROUND_FEATURE_SAVE_PATH)
        print('finished background_features read')

        # label {movie:label}
        self.label = {}
        label_path = cfg.DATA.VISR_LABEL
        label_file = open(label_path)
        label_lines = label_file.readlines()
        for line in label_lines:
            movie_name = line.split('\t')[0].split('.')[0]
            label = line.split('\t')[1].strip()
            self.label[movie_name] = str(int(label) + 1)  # label shift outside 0
        print('finished label read')

    def __len__(self):
        return len(self.video_scene_name)

    def __getitem__(self, idx):
        end = False
        video_scene_name = self.video_scene_name[idx]
        video_name = video_scene_name.split('_')[0]
        scene_id = int(video_scene_name.split('_')[1])
        if idx == len(self.video_scene_name) - 1:
            end = True
        elif self.video_scene_name[idx + 1].split('_')[0] != video_name:
            end = True

        bert_features = self.bert_features[video_name]

        background_feature = self.background_features[video_name]

        face_features = {}
        face_features = self.face_features[int(video_name)][scene_id]
        
        graph_features_face_dim, graph_feature_edge_idx, adj = self.construct_graph_feature(face_features)

        label = []  # if no relation annotation in label_file, then assign 0
        for edge_idx in graph_feature_edge_idx:
            label.append(torch.tensor(int(self.label[video_name]), dtype=torch.long))

        bert_features = self.construct_bert_feature(bert_features, graph_feature_edge_idx)
        graph_feature_edge_idx = torch.tensor(graph_feature_edge_idx, dtype=torch.long)
        label = torch.stack(label)
        label = label.reshape(label.shape[0], 1)

        return bert_features, background_feature, graph_features_face_dim, graph_feature_edge_idx, adj, label, end

    # select related bert_feature from overall bert_features
    def construct_bert_feature(self, bert_feature, graph_feature_edge_idx):
        bert_features = []
        for i in graph_feature_edge_idx:
            bert_features.append(bert_feature)
        return torch.stack(bert_features)

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

        return torch.stack(graph_features).squeeze(1), graph_feature_edge_idx, adj