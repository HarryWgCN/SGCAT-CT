import json
import os

import torch

from config.defaults import cfg

scene_graph_dir = ''
scene_graph_dict = {}

for path, dirnames, filenames in os.walk(scene_graph_dir):
    for filename in filenames:
        movie_scene = filename[len('scene_knowledge_graph'):].split('.')[0]
        movie_name = movie_scene.split('-')[0]
        scene_id = movie_scene.split('-')[1]
        if movie_name not in scene_graph_dict:
            scene_graph_dict[movie_name] = {}
        if scene_id not in scene_graph_dict[movie_name]:
            scene_graph_dict[movie_name][scene_id] = {}
        file_path = os.path.join(scene_graph_dir, filename)
        scene_graph_file = open(file_path, 'r')
        scene_graph = json.load(scene_graph_file)
        temp_key_to_text = {}
        for record in scene_graph['nodes']:
            temp_key_to_text[record['key']] = record['text']
        for record in scene_graph['links']:
            if 'text' in record:
                source = temp_key_to_text[record['from']]
                target = temp_key_to_text[record['to']]
                scene_graph_dict[movie_name][scene_id] = (source, target, record['text'])
                
# entity_type {movie:{entity:type}}
# label {movie:{character_pair:label}}, character_pair:(entity_1, entity_2) entity_1 < entity_2
label = {}
entity_type = {}
id_in_file_to_character = {}
# relation_mapping = {"spouse of":"Family","parent of":"Family","extended family of":"Family","divorced to":"Family","descendant of":"Family","sibling of":"Family","aunt/uncle":"Family","spouse":"Family","engaged":"Family","engages with":"Family","parent":"Family","cousin":"Family","sibling":"Family","collaborates with":"Colleague","employee of":"Colleague","supervisor of":"Colleague","teacher of":"Colleague","mentor of":"Colleague","ex partner of":"Colleague","colleague of":"Colleague","collaborator":"Colleague","colleague":"Colleague","controlled by":"Colleague","responsible for":"Colleague","owns":"Colleague","influences":"Acquaintance","in relationship with":"Acquaintance","has met":"Acquaintance","knows of":"Acquaintance","knows in passing":"Acquaintance","knows in passing，acquaintance of":"Acquaintance","would like to know":"Acquaintance","lost contact with":"Acquaintance","ambivalent of":"Opponent","bullies":"Opponent","antagonist of":"Opponent","dislikes":"Opponent","enemy of":"Opponent","customer of":"Service","patient of":"Service","takes care of":"Service","doctor of":"Service","friend of":"Friend","friend":"Friend"}
# relation_to_id = {'Others':0,'Family':1, 'Colleague':2, 'Acquaintance':3, 'Opponent':4, 'Service':5, 'Friend':6}
for path, dirnames, filenames in os.walk(cfg.DATA.HLVU_LABEL_DIR):
    for filename in filenames:
        if len(filename.split('.')) > 2:
            entity_type_file_path = os.path.join(cfg.DATA.HLVU_LABEL_DIR, filename)
            entity_type_file = open(entity_type_file_path)
            lines = entity_type_file.readlines()
            entity_type[filename.split('.')[0]] = {}
            for line in lines:
                entity_type[filename.split('.')[0]][line.split(':')[0].strip().lower()] = line.split(':')[1].strip()
        else:
            entity_type_file_path = os.path.join(cfg.DATA.HLVU_LABEL_DIR, filename)
            entity_type_file = open(entity_type_file_path)
            lines = entity_type_file.readlines()
            label[filename.split('.')[0]] = {}
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
                    label[filename.split('.')[0]][character_pair] = line[prefix:].strip().lower()
                else:
                    id_in_file = line.split(' ')[0]
                    character_name = line[len(id_in_file):].strip().lower()
                    id_in_file_to_character[filename.split('.')[0]][id_in_file] = character_name
print('end')
