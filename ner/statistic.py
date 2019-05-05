def external_dict(exdict, test_file):
    entities = []
    with open(test_file, 'r', encoding='utf-8') as fin:
        entity = ''
        for line in fin:
            if line != '\n':
                info = line.strip().split('\t')
                char, label = info[0], info[-1]
                if label.startswith('B-'):
                    if len(entity) > 0:
                        entities.append(entity)
                        entity = ""
                    assert len(entity) == 0
                    entity += char
                elif label.startswith('I-'):
                    assert len(entity) >= 1
                    entity += char
                elif label == 'O':
                    if len(entity) > 0:
                        entities.append(entity)
                        entity = ""
                else:
                    raise RuntimeWarning("label oov")
            else:
                if len(entity) > 0:
                    entities.append(entity)
                    entity = ""
    oov = list(set(entities).difference(exdict))
    match = list(set(entities).intersection(exdict))
    oov_num,match_num,total = 0,0,0
    ori,external = [],[]
    for e in entities:
        if e in oov:
            oov_num+=1
        if e in match:
            match_num += 1
            ori.append(e)
    for e in entities:
        if total+match_num < 10000:
            external.append(e)
            total+=1
    print("dict num:{}\nentities num:{}\noov num:{}\nmatch num:{}\n".format(len(exdict),len(entities),oov_num,match_num))
    ori.extend(external)
    return ori

