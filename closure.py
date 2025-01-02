def getRelation(dataset):
    #获取有特殊性质的relations
    with open(f'meta_dataset/{dataset}/TBoxtriples.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            a, r, b = line.split('!==!')
            if a == 'TRANSITIVE':
                with open(f'meta_dataset/{dataset}/transitive.txt', 'w') as f:
                    f.write(r + '\n')
            elif a == 'SYMMETRIC':
                with open(f'meta_dataset/{dataset}/symmetric.txt', 'w') as f:
                    f.write(r + '\n')
            elif a == 'REFLEXIVE':
                with open(f'meta_dataset/{dataset}/reflexive.txt', 'w') as f:
                    f.write(r + '\n')

def symmetric_closure(dataset):
    ## 可能会有重复的，后边需要去重一下
    f_symmetric = open(f'meta_dataset/{dataset}/symmetric_closure.txt', 'w')
    symmetric_relations = set()
    with open(f'meta_dataset/{dataset}/symmetric.txt', 'r') as f:
        for line in f.readlines():
            symmetric_relations.add(line.strip())
    with open(f'meta_dataset/{dataset}/ABoxtriples.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            a, r, b = line.split('!==!')
            if r in symmetric_relations:
                f_symmetric.write(b + '!==!' + r + '!==!' + a + '\n')
    f_symmetric.close()

def transitive_closure(dataset):

    def closure(triples:set, r):
        total_new_triples = set()
        changed = True
        while changed:
            changed = False
            new_triples = set()
            for a, _, b in triples:
                for b2, _, c in triples:
                    if b == b2 and (a, r, c) not in triples:
                        new_triples.add((a, r, c))
                        changed = True
            triples.update(new_triples)
            total_new_triples.update(new_triples)
        return total_new_triples

    f_transitive = open(f'meta_dataset/{dataset}/transitive_closure.txt', 'w')
    transitive_relations = set()
    with open(f'meta_dataset/{dataset}/transitive.txt', 'r') as f:
        for line in f.readlines():
            transitive_relations.add(line.strip())
    for relation in transitive_relations:
        triples = set()
        with open(f'meta_dataset/{dataset}/ABoxtriples.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                a, r, b = line.split('!==!')
                if r == relation:
                    triples.add((a,r,b))
        
        total_new_triples = closure(triples, relation)
        for triple in total_new_triples:
            a, r, b = triple
            f_transitive.write(a + '!==!' + r + '!==!' + b + '\n')
    f_transitive.close()

transitive_closure('YAGO3-10')