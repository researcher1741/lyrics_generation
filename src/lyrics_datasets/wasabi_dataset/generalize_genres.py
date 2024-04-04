from collections import defaultdict
import jsonlines
from tqdm import tqdm
from collections import Counter
def generalize_genres(path, outpath):
    genres = set()
    genre_counter = Counter()
    with open(path) as lines:
        for line in tqdm(lines):
            count, *genre = line.strip().split(' ')
            genre = ' '.join(genre)
            genre = genre.replace('\u200f', '').replace('\u200e', '')
            genres.add(genre)
            genre_counter[genre] += int(count)
    
    groups = defaultdict(set)
    for g in genres:
        for g1 in genres:
            if g == g1:
                continue
            if g.lower() in g1.lower() or g1.lower() in g.lower():
                if len(g) < len(g1):
                    groups[g].add(g1)
                else:
                    groups[g1].add(g)
    reversed_groups = defaultdict(Counter)
    most_common = {k for k, _ in genre_counter.most_common(100)}
    not_covered = set()
    for k, v in groups.items():
        for x in v:
            count = genre_counter[k]
            reversed_groups[x][k] = count

    not_covered_items = 0
    mapping = dict()
    for k, v in reversed_groups.items():
        if all([x not in most_common for x in v]):
            not_covered.add(k)
            not_covered_items += genre_counter[k]
            mapping[k] = 'Other'

        else:
            k_most_common = [x for x in v if x in most_common]
            if len(k_most_common) > 1: 
                max_idx = -1
                best_candidate = None
                for x in k_most_common:
                    idx = k.index(x)
                    if max_idx < idx:
                        max_idx = idx
                        best_candidate = x
                mapping[k] = best_candidate + '\t**\t' + str(k_most_common)
            else:
                mapping[k] = k_most_common[0]
    with open(outpath, 'w') as writer:
        for k, v in mapping.items():
            writer.write(f'{k}\t{v}\n')
        


if __name__ == '__main__':
    path  = 'data/genres.counter.txt'
    outpath = 'data/genres_mapping_100.txt'
    generalize_genres(path, outpath)