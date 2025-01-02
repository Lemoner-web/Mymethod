dataset = 'RadLex'

entities = set()
relations = set()
concepts = set()
with open(f'meta_dataset/{dataset}/concept_hierarchy.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        A, B = line.split('\t')
        concepts.add(A)
        concepts.add(B)

with open(f'meta_dataset/{dataset}/individual_to_concept.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        a, A = line.split('\t')
        entities.add(a)
        concepts.add(A)

with open(f'meta_dataset/{dataset}/concept_relations.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        A, r, B = line.split('\t')
        concepts.add(A)
        relations.add(r)
        concepts.add(B)

with open(f'meta_dataset/{dataset}/individual_relations.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        a, r, b = line.split('\t')
        entities.add(a)
        relations.add(r)
        entities.add(b)

entities_dict = {entity:str(idx) for idx, entity in enumerate(entities)}
concepts_dict = {concept:str(idx) for idx, concept in enumerate(concepts)}
relations_dict = {relation:str(idx) for idx, relation in enumerate(relations)}

# write dict
with open(f'dataset/{dataset}/entities.txt', 'w') as f:
    for key, value in entities_dict.items():
        f.write(key + '\t' + value + '\n')

with open(f'dataset/{dataset}/concepts.txt', 'w') as f:
    for key, value in concepts_dict.items():
        f.write(key + '\t' + value + '\n')

with open(f'dataset/{dataset}/relations.txt', 'w') as f:
    for key, value in relations_dict.items():
        f.write(key + '\t' + value + '\n')

# write index based data
with open(f'meta_dataset/{dataset}/concept_hierarchy.txt', 'r') as f:
    f1 = open(f'dataset/{dataset}/concept_hierarchy.txt', 'w')
    for line in f.readlines():
        line = line.strip()
        A, B = line.split('\t')
        f1.write(concepts_dict[A] + '\t' + concepts_dict[B] + '\n')
    f1.close()

with open(f'meta_dataset/{dataset}/individual_to_concept.txt', 'r') as f:
    f1 = open(f'dataset/{dataset}/individual_to_concept.txt', 'w')
    for line in f.readlines():
        line = line.strip()
        a, A = line.split('\t')
        f1.write(entities_dict[a] + '\t' + concepts_dict[A] + '\n')
    f1.close()

with open(f'meta_dataset/{dataset}/concept_relations.txt', 'r') as f:
    f1 = open(f'dataset/{dataset}/concept_relations.txt', 'w')
    for line in f.readlines():
        line = line.strip()
        A, r, B = line.split('\t')
        f1.write(concepts_dict[A] + '\t' + relations_dict[r] + '\t' + concepts_dict[B] + '\n')
    f1.close()

with open(f'meta_dataset/{dataset}/individual_relations.txt', 'r') as f:
    f1 = open(f'dataset/{dataset}/individual_relations.txt', 'w')
    for line in f.readlines():
        line = line.strip()
        a, r, b = line.split('\t')
        f1.write(entities_dict[a] + '\t' + relations_dict[r] + '\t' + entities_dict[b] + '\n')
    f1.close()

with open(f'meta_dataset/{dataset}/valid.txt', 'r') as f:
    f1 = open(f'dataset/{dataset}/valid.txt', 'w')
    for line in f.readlines():
        line = line.strip()
        a, r, b = line.split('\t')
        f1.write(entities_dict[a] + '\t' + relations_dict[r] + '\t' + entities_dict[b] + '\n')
    f1.close()

with open(f'meta_dataset/{dataset}/test.txt', 'r') as f:
    f1 = open(f'dataset/{dataset}/test.txt', 'w')
    for line in f.readlines():
        line = line.strip()
        a, r, b = line.split('\t')
        f1.write(entities_dict[a] + '\t' + relations_dict[r] + '\t' + entities_dict[b] + '\n')
    f1.close()

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