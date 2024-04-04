import jsonlines
from sklearn.utils import shuffle
from tqdm import tqdm

def get_key(song):
    return song['artist'].lower() + '_' + song['title'].lower()

def create_test_and_dev(training_path, bugged_test_path, bugged_dev_path, all_path, new_test_path, new_dev_path):
    training_keys = set()
    with jsonlines.open(training_path) as lines:
        for line in tqdm(lines, desc='loading training keys'):
            key = get_key(line)
            training_keys.add(key)
    test_keys = set()
    test_lines = list()
    with jsonlines.open(bugged_test_path) as lines:
        for line in tqdm(lines, desc='reading test'):
            key = get_key(line)
            if key not in training_keys:
                test_keys.add(key)
                test_lines.append(line)
    dev_keys = set()
    dev_lines = list()
    with jsonlines.open(bugged_dev_path) as lines:
        for line in tqdm(lines, desc='reading dev'):
            key = get_key(line)
            if key not in training_keys and key not in test_keys:
                dev_keys.add(key)
                dev_lines.append(line)
    all_keys = set()
    all_lines = []
    with jsonlines.open(all_path) as lines:
        for line in tqdm(lines, desc='reading all songs'):
            key = get_key(line)
            schema = line["rhyming_schema"]
            if len(schema) > 10:
                continue 
            if len(schema) == len(set(schema)):
                continue
            if key not in training_keys and key not in dev_keys and key not in test_keys:
                all_keys.add(key)
                all_lines.append(line)
    shuffle(all_lines)
    dataset_size = 3500
    new_dev_lines = all_lines[:dataset_size - len(dev_lines)]
    all_lines = all_lines[dataset_size - len(dev_lines):]
    new_test_lines = all_lines[:dataset_size - len(test_lines)]
    dev_lines = dev_lines + new_dev_lines
    test_lines = test_lines + new_test_lines

    with jsonlines.open(new_test_path, 'w') as writer:
        shuffle(test_lines)
        for line in tqdm(test_lines, desc='writing test'):
            if 'lang' not in line:
                line['lang'] = 'english'
            writer.write(line) 
    
    with jsonlines.open(new_dev_path, 'w') as writer:
        shuffle(dev_lines)
        for line in tqdm(dev_lines, desc='writing dev'):
            if 'lang' not in line:
                line['lang'] = 'english'
            writer.write(line) 
            
    
if __name__ == '__main__':
    training_path='./LG/DATA/genius_section_0.2/train.jsonl'
    bugged_test_path = './LG/DATA/genius_section_0.2/test.bugged.jsonl'
    bugged_dev_path = './LG/DATA/genius_section_0.2/dev.bugged.jsonl'
    all_path = 'data/genius/phonemised_dataset/section_dataset.jsonl'
    new_test_path = './LG/DATA/genius_section_0.2/test.3500.jsonl'
    new_dev_path = './LG/DATA/genius_section_0.2/dev.3500.jsonl'
    create_test_and_dev(training_path, bugged_test_path, bugged_dev_path, all_path, new_test_path, new_dev_path)



    