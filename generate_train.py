import os

def getName(signature:str):
    # <http://radlex.org/RID/Part_Of> -> Part_Of
    # <http://example.com/YAGO3-10/http://www.falkirk.gov.uk> -> http://www.falkirk.gov.uk
    if 'YAGO3-10' in signature:
        return signature[signature.index('YAGO3-10') + 9 : -1]
    if signature.startswith('<') and signature.endswith('>'):
        return signature.split('/')[-1][:-1]

def concept_relations(dataset):
    # generate (A,r,B) from TBox
    line_set = set()
    f_write = open(f'meta_dataset/{dataset}/concept_relations.txt', 'w')
    with open(f'meta_dataset/{dataset}/TBoxtriples.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if 'owl:' in line: continue
            if line in line_set:
                continue
            A, r, B = line.split('!==!')
            if r.startswith('<') and r.endswith('>') and A.startswith('<') and A.endswith('>'):
                f_write.write(getName(A)+'\t'+getName(r)+'\t'+getName(B)+'\n')
                line_set.add(line)
    f_write.close()

def individual_relations(dataset):
    # generate (a,r,b) from ABox, *_closure
    # need to remove triples in (valid or test)

    need_remove_triples = set()
    with open(f'meta_dataset/{dataset}/valid.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            a,r,b = line.split('\t')
            need_remove_triples.add((a,r,b))

    with open(f'meta_dataset/{dataset}/test.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            a,r,b = line.split('\t')
            need_remove_triples.add((a,r,b))

    line_set = set()
    f_write = open(f'meta_dataset/{dataset}/individual_relations.txt', 'w')
    files = [f'meta_dataset/{dataset}/ABoxtriples.txt',
             f'meta_dataset/{dataset}/symmetric_closure.txt',
             f'meta_dataset/{dataset}/transitive_closure.txt']
    for file_path in files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if 'owl:' in line: continue
                    if line in line_set:
                        continue
                    a, r, b = line.split('!==!')
                    if r.startswith('<') and r.endswith('>') and a.startswith('<') and a.endswith('>'):
                        triple = (getName(a), getName(r), getName(b))
                        if triple in need_remove_triples: continue
                        f_write.write('\t'.join(triple) + '\n')
                        line_set.add(line)
    f_write.close()

def individual_to_concept(dataset):
    # generate A(a) from ABox and domain_range_triples
    line_set = set()
    f_write = open(f'meta_dataset/{dataset}/individual_to_concept.txt', 'w')
    files = [f'meta_dataset/{dataset}/ABoxtriples.txt',
             f'meta_dataset/{dataset}/domain_range_triples.txt'
             ]
    for file_path in files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if 'owl:' in line: continue
                    if line in line_set:
                        continue
                    a, r, A = line.split('!==!')
                    if r == 'TYPE':
                        f_write.write(getName(a)+'\t'+getName(A)+'\n')
                        line_set.add(line)
    f_write.close()

def concept_hierarchy(dataset):
    # generate A subclassof B from TBox
    line_set = set()
    f_write = open(f'meta_dataset/{dataset}/concept_hierarchy.txt', 'w')
    with open(f'meta_dataset/{dataset}/TBoxtriples.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if 'owl:' in line: continue
            if line in line_set:
                continue
            A, r, B = line.split('!==!')
            if r == 'SUBCLASSOF':
                f_write.write(getName(A)+'\t'+getName(B)+'\n')
                line_set.add(line)
    f_write.close()

concept_relations('YAGO3-10')
individual_relations('YAGO3-10')
individual_to_concept('YAGO3-10')   
concept_hierarchy('YAGO3-10')