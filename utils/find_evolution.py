import os

from config.defaults import cfg

now_scene = ''
label_dir = cfg.DATA.LABEL_DIR
for path, dirnames, filenames in os.walk(label_dir):
    for filename in filenames:
        label_dict = {}
        now_movie = filename[:9]
        if now_movie not in label_dict:
            label_dict[now_movie] = {}
        label_file = open(os.path.join(label_dir, filename), 'r')
        for line in label_file.readlines():
            if line == '\n':
                continue
            elif line[:5] == 'scene':
                now_scene = int(line[6:9])
            else:
                character_0 = line.split(';')[0]
                character_1 = line.split(';')[1]
                character_pair = (character_0, character_1)
                label = int(line.split(';')[2]) + 1
                if character_pair in label_dict:
                    if label_dict[character_pair][1] != label:
                        print(f'found one {now_movie}, {character_0} and {character_1} at {now_scene}, relation {label}; former at {label_dict[character_pair][0]}, relation {label_dict[character_pair][1]}')
                label_dict[character_pair] = (now_scene, label)
