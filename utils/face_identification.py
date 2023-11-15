import torch


def get_face_name(scene_face_dict, scene_id, face):
    character_num = 0
    for key, value in scene_face_dict.items():
        for key_, value_ in value.items():
            character_num += 1
            face_similarity = torch.cosine_similarity(value_, face, dim=0)
            if face_similarity >= 0.80:
                return key_
    return character_num
