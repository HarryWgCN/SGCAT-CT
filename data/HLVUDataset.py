import json
import os
import random

import torch
from torch.utils.data import Dataset
from config.defaults import cfg
from itertools import permutations
import numpy as np

class HLVUDataset(Dataset):
    def __init__(self, train):
        self.character_to_idx = {}  # character_name - idx
        self.idx_to_character = {}  # idx - character_name
        self.edge_to_idx = {}  # edge(node1_idx, node2_idx) - idx, here node1_idx < node2_idx
        self.idx_to_edge = {}  # idx - edge(node1_idx, node2_idx)

        # video-clip-names [a]
        data_file = open(cfg.DATA.HLVU_SPLIT_PATH, 'r')
        data_json_object = json.load(data_file)
        if train:
            self.video_scene_name = data_json_object['train']
        else:
            self.video_scene_name = data_json_object['test']

        # add scene  movie_scene_frame
        temp_video_scene_name = {}
        temp_video_scene_list = []
        if os.path.exists(cfg.MODEL.HLVU_VIDEO_SCENE_NAME_TRAIN_PATH) and train:
            temp_video_scene_list = torch.load(cfg.MODEL.HLVU_VIDEO_SCENE_NAME_TRAIN_PATH)
        if os.path.exists(cfg.MODEL.HLVU_VIDEO_SCENE_NAME_TEST_PATH) and not train:
            temp_video_scene_list = torch.load(cfg.MODEL.HLVU_VIDEO_SCENE_NAME_TEST_PATH)
        for record in self.video_scene_name:
            if len(temp_video_scene_list) > 0 and len(temp_video_scene_name) == 0:
                break
            video_dir = os.path.join(cfg.DATA.HLVU_FRAME_DIR, record)
            for path, dirnames, filenames in os.walk(video_dir):
                for filename in filenames:
                    if record not in temp_video_scene_name:
                        temp_video_scene_name[record] = []
                    temp_video_scene_name[record].append(f'{record}_{filename.split("_")[0]}_{filename.split(".")[0].split("_")[1]}')
        for key, value in temp_video_scene_name.items():
            if len(temp_video_scene_list) > 0 and len(temp_video_scene_name) == 0:
                break
            value.sort(key=lambda x: int(x.split('_')[1]))
            temp_video_scene_list += value
        self.video_scene_name = temp_video_scene_list
        if train:
            torch.save(self.video_scene_name, cfg.MODEL.HLVU_VIDEO_SCENE_NAME_TRAIN_PATH)
        if not train:
            torch.save(self.video_scene_name, cfg.MODEL.HLVU_VIDEO_SCENE_NAME_TEST_PATH)

        # entity_type {movie:{entity:type}}
        # label {movie:{character_pair:label}}, character_pair:(entity_1, entity_2) entity_1 < entity_2
        self.label = {}
        label = {}
        self.entity_type = {}
        id_in_file_to_character = {}
        self.relation_to_index = {}
        relation_mapping = {"spouse of":"Family","parent of":"Family","extended family of":"Family","divorced to":"Family","descendant of":"Family","sibling of":"Family","aunt/uncle":"Family","spouse":"Family","engaged":"Family","engages with":"Family","parent":"Family","cousin":"Family","sibling":"Family","collaborates with":"Colleague","employee of":"Colleague","supervisor of":"Colleague","teacher of":"Colleague","mentor of":"Colleague","ex partner of":"Colleague","colleague of":"Colleague","collaborator":"Colleague","colleague":"Colleague","controlled by":"Colleague","responsible for":"Colleague","owns":"Colleague","influences":"Acquaintance","in relationship with":"Acquaintance","has met":"Acquaintance","knows of":"Acquaintance","knows in passing":"Acquaintance","knows in passing，acquaintance of":"Acquaintance","would like to know":"Acquaintance","lost contact with":"Acquaintance","ambivalent of":"Opponent","bullies":"Opponent","antagonist of":"Opponent","dislikes":"Opponent","enemy of":"Opponent","customer of":"Service","patient of":"Service","takes care of":"Service","doctor of":"Service","friend of":"Friend","friend":"Friend"}
        relation_to_id = {'Others':0,'Family':1, 'Colleague':2, 'Acquaintance':3, 'Opponent':4, 'Service':5, 'Friend':6}
        for path, dirnames, filenames in os.walk(cfg.DATA.HLVU_LABEL_DIR):
            for filename in filenames:
                if len(filename.split('.')) > 2:
                    entity_type_file_path = os.path.join(cfg.DATA.HLVU_LABEL_DIR, filename)
                    entity_type_file = open(entity_type_file_path)
                    lines = entity_type_file.readlines()
                    self.entity_type[filename.split('.')[0]] = {}
                    for line in lines:
                        self.entity_type[filename.split('.')[0]][line.split(':')[0].strip().lower()] = line.split(':')[1].strip()
                else:
                    entity_type_file_path = os.path.join(cfg.DATA.HLVU_LABEL_DIR, filename)
                    entity_type_file = open(entity_type_file_path)
                    lines = entity_type_file.readlines()
                    label[filename.split('.')[0]] = {}
                    self.character_to_idx[filename.split('.')[0]] = {}
                    self.idx_to_character[filename.split('.')[0]] = {}
                    relation = False
                    id_in_file_to_character[filename.split('.')[0]] = {}
                    for line in lines:
                        if line == '#\n':
                            relation = True
                            continue
                        if relation:
                            source = id_in_file_to_character[filename.split('.')[0]][line.split(' ')[0]].lower()
                            target = id_in_file_to_character[filename.split('.')[0]][line.split(' ')[1].strip()].lower()
                            character_pair = (source, target) if source < target else (target, source)
                            prefix = len(line.split(' ')[0]) + 2 + len(line.split(' ')[1])
                            if line[prefix:].strip().lower() in relation_mapping:
                                relation = relation_mapping[line[prefix:].strip().lower()]
                            else:
                                relation = 'Others'
                            label[filename.split('.')[0]][character_pair] = relation_to_id[relation]
                        else:
                            id_in_file = line.split(' ')[0]
                            character_name = line[len(id_in_file):].strip().lower()
                            id_in_file_to_character[filename.split('.')[0]][id_in_file] = character_name
                            if character_name not in self.character_to_idx[filename.split('.')[0]]:
                                idx = len(self.character_to_idx[filename.split('.')[0]])
                                self.character_to_idx[filename.split('.')[0]][character_name] = idx
                                self.idx_to_character[filename.split('.')[0]][idx] = character_name
        for key, value in label.items():
            video_name = key
            if video_name not in self.label:
                self.label[video_name] = {}
            for key_, value_ in value.items():
                source_id = self.character_to_idx[video_name][key_[0]]
                target_id = self.character_to_idx[video_name][key_[1]]
                character_pair = (source_id, target_id) if source_id < target_id else (target_id, source_id)
                self.label[video_name][character_pair] = value_

        # for key, value in self.label.items():
        #     for key_, value_ in self.label[key].items():
        #         if self.entity_type[key][key_[0]] == 'Person' and self.entity_type[key][key_[1]] == 'Person':
        #             if value_.lower() not in self.relation_to_index:
        #                 self.relation_to_index[value_.lower()] = len(self.relation_to_index)

        # face-features {movie:entity:tensor}


        self.face_features = {}
        if os.path.exists(cfg.MODEL.HLVU_FACE_FEATURE_SAVE_PATH):
            self.face_features = torch.load(cfg.MODEL.HLVU_FACE_FEATURE_SAVE_PATH)
        face_feature_dir = cfg.FEATURE.HLVU_FACE_FEATURE_DIR
        for path, dirnames, filenames in os.walk(face_feature_dir):
            if len(self.face_features) > 0:
                break
            for filename in filenames:
                face_feature_file_path = os.path.join(face_feature_dir, filename)
                face_fature_json_object = torch.load(face_feature_file_path)
                self.face_features[filename.split('.')[0]] = {}
                for key, value in face_fature_json_object.items():
                    video_name = key.split('_')[0]
                    entity_name = key[(len(key.split('_')[0]) + 1):].split('.')[0]
                    if entity_name.lower().split('_')[0].rstrip('0123456789') not in self.entity_type[video_name]:
                        continue
                    if self.entity_type[video_name][entity_name.lower().split('_')[0].rstrip('0123456789')] == 'Person':
                        self.face_features[video_name][self.character_to_idx[video_name][entity_name.lower().split('_')[0].rstrip('0123456789')]] = torch.Tensor(value[0])
            torch.save(self.face_features, cfg.MODEL.HLVU_FACE_FEATURE_SAVE_PATH)
        print('finished face_feature read')

        # bert-features {movie:{scene:{edge_key: tensor}}}, edge_key: (character_0_idx, character_1_idx) character_0_idx < character_1_idx
        self.bert_features = {}
        if os.path.exists(cfg.MODEL.HLVU_BERT_FEATURE_SAVE_PATH):
            self.bert_features = torch.load(cfg.MODEL.HLVU_BERT_FEATURE_SAVE_PATH)
        if not len(self.bert_features) > 0:
            bert_key_to_video_key = {'time_expired':'TimeExpired', 'SuperHero':'superHero', 'ChainedforLife':'ChainedForLife', 'losing_ground':'losingGround', 'Huckleberry_Finn':'huckleberryFinn', 'Liberty_Kid':'LibertyKid', 'Road_To_Bali':'RoadToBali', 'The_Big_Something':'TheBigSomething', 'Manos':'Manos', 'Nuclear_Family':'nuclearFamily', 'honey':'honey', 'Bagman':'Bagman', 'shooters':'Shooter', 'The_Illusionist':'TheIllusionist', 'sophie':'Sophie', 'like_me':'likeMe', 'Calloused_Hands':'callousedHands', 'spiritual_contact':'spiritualContact', 'Valkaama':'Valkaama'}
            bert_feature_object = torch.load(cfg.FEATURE.HLVU_BERT_FEATURE_PATH)
            for key, value in bert_feature_object.items():
                self.bert_features[bert_key_to_video_key[key]] = {}
                for key_, value_ in value.items():
                    self.bert_features[bert_key_to_video_key[key]][key_] = torch.tensor(np.mean(value_, axis=0))
            torch.save(self.bert_features, cfg.MODEL.HLVU_BERT_FEATURE_SAVE_PATH)
        print('finished bert_feature read')

        # background-features {movie:{scene:tensor}}
        self.background_features = {}
        if os.path.exists(cfg.MODEL.HLVU_BACKGROUND_FEATURE_SAVE_PATH):
            self.background_features = torch.load(cfg.MODEL.HLVU_BACKGROUND_FEATURE_SAVE_PATH)
        background_feature_dir = cfg.FEATURE.HLVU_BACKRGOUND_FEATURE_DIR
        for path, dirnames, filenames in os.walk(background_feature_dir):
            if len(self.background_features) > 0:
                break
            for filename in filenames:
                file_path = os.path.join(background_feature_dir, filename)
                background_feature_json_object = torch.load(file_path)
                self.background_features[filename.split('.')[0]] = {}
                for key, value in background_feature_json_object.items():
                    self.background_features[filename.split('.')[0]][key] = []
                    for key_, value_ in background_feature_json_object[key].items():
                        self.background_features[filename.split('.')[0]][key].append(value_)
            for key, value in self.background_features.items():
                for key_, value_ in self.background_features[key].items():
                    temp_numpy_array = np.array(value_)
                    temp_numpy_array = np.mean(temp_numpy_array, axis=0)
                    self.background_features[key][key_] = torch.tensor(temp_numpy_array)
            torch.save(self.background_features, cfg.MODEL.HLVU_BACKGROUND_FEATURE_SAVE_PATH)
        print('finished background_features read')
        count = 0


    def __len__(self):
        return len(self.video_scene_name)

    def __getitem__(self, idx):
        end = False
        video_scene_name = self.video_scene_name[idx]
        video_name = video_scene_name.split('_')[0]
        scene_id = video_scene_name.split('_')[1]
        if idx == len(self.video_scene_name) - 1:
            end = True
        elif self.video_scene_name[idx + 1].split('_')[0] != video_name:
            end = True
        # try adjacent scenes when not found

        if scene_id in self.bert_features[video_name]:
            bert_features = self.bert_features[video_name][scene_id]
        else:
            bert_features = torch.zeros(size=(1, cfg.FEATURE.TEXT_DIM))

        video_name_temp = video_name
        if video_name == 'ChainedForLife':
            video_name_temp = 'ChainedforLife'
        background_feature = self.background_features[video_name_temp][scene_id]

        # if not found face_feature, find face_features in neighboring scenes
        face_features = []
        face_features = self.face_features[video_name]
        graph_features_face_dim, graph_feature_edge_idx, adj = self.construct_graph_feature(face_features)

        label = []  # if no relation annotation in label_file, then assign 0
        for edge_idx in graph_feature_edge_idx:
            edge_key = self.idx_to_edge[edge_idx]
            if edge_key not in self.label[video_name]:
                label.append(torch.tensor(random.randint(1, cfg.MODEL.NUM_CLASSES-1), dtype=torch.long))
            else:
                label.append(torch.tensor(self.label[video_name][edge_key], dtype=torch.long))

        bert_features = self.construct_bert_feature(bert_features, graph_feature_edge_idx)
        graph_feature_edge_idx = torch.tensor(graph_feature_edge_idx, dtype=torch.long)
        label = torch.stack(label)
        label = label.reshape(label.shape[0], 1)

        return bert_features, background_feature, graph_features_face_dim, graph_feature_edge_idx, adj, label, end

    # select related bert_feature from overall bert_features
    def construct_bert_feature(self, bert_features, graph_feature_edge_idx):
        bert_feature = []
        for i in graph_feature_edge_idx:
            bert_feature.append(bert_features)
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
        # except:
        #     print('hi')
        #     return torch.zeros(size=(1, 1)), [0], 1
