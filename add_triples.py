## add triples with domain and range
## 需要排除掉bot和top
## 这里先不做处理，考虑在最后统一处理
import os

def getRelation(dataset):
    #获取有domain range定义的relations
    with open(f'meta_dataset/{dataset}/TBoxtriples.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            a, r, b = line.split('!==!')
            if a == 'DOMAIN':
                with open(f'meta_dataset/{dataset}/domain.txt', 'w') as f:
                    f.write(r + '!==!' + b + '\n')
            elif a == 'RANGE':
                with open(f'meta_dataset/{dataset}/range.txt', 'w') as f:
                    f.write(r + '!==!' + b + '\n')

def addTriples(dataset):
    
    relation_map = {} ## {r1 : {domain:, range:}, r2:}
    with open(f'meta_dataset/{dataset}/domain.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            r, domain_ = line.split('!==!')
            if r not in relation_map:
                relation_map[r] = {'domain': domain_, 'range':''}
    with open(f'meta_dataset/{dataset}/range.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            r, range_ = line.split('!==!')
            if r not in relation_map:
                relation_map[r] = {'domain': '', 'range':range_}
            else:
                relation_map[r]['range'] = range_

    ABoxPath = f'meta_dataset/{dataset}/ABoxtriples.txt'
    closureABoxPaths = [f'meta_dataset/{dataset}/symmetric_closure.txt', f'meta_dataset/{dataset}/transitive_closure.txt']
    add_triples = set()
    for file in [ABoxPath] + closureABoxPaths:
        if os.path.exists(file):
            with open(file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    a, r, b = line.split('!==!')
                    if r in relation_map.keys():
                        domain_ = relation_map[r]['domain']
                        range_ = relation_map[r]['range']
                        if domain_ != '': add_triples.add((a, domain_))
                        if range_ != '': add_triples.add((b, range_))
    return add_triples


dataset = 'YAGO3-10'
new_triples = addTriples(dataset)
with open(f'meta_dataset/{dataset}/domain_range_triples.txt', 'w') as f:
    for triple in new_triples:
        entity, restriction = triple
        f.write(entity + '!==!' + 'TYPE' + '!==!' + restriction + '\n')