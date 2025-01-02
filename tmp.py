dataset = 'RadLex'

entities_dict = {}
relations_dict = {}
with open(f'dataset/{dataset}/entities.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        entity, idx = line.split('\t')
        entities_dict[entity] = idx

with open(f'dataset/{dataset}/relations.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        relation, idx = line.split('\t')
        relations_dict[relation] = idx

with open(f'meta_dataset/{dataset}/train.txt', 'r') as f:
    f1 = open(f'dataset/{dataset}/train.txt', 'w')
    for line in f.readlines():
        line = line.strip()
        a, r, b = line.split('\t')
        try:    
            f1.write(entities_dict[a] + '\t' + relations_dict[r] + '\t' + entities_dict[b] + '\n')
        except:
            pass
    f1.close()