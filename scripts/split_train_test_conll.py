# coding=utf-8

with open("", encoding="utf-8") as f:
    sentences = []
    words = []
    labels = []
    docs = []
    for line in f:
        contends = line.strip()
        tokens = contends.split(' ')
        if contends.startswith("-DOCSTART-"):
            if len(sentences) > 0:
                docs.append(sentences)
                sentences = []
            continue
        if len(tokens) == 2:
            words.append(tokens[0])
            labels.append(tokens[-1])
        else:
            if len(contends) == 0 and len(words) > 0:
                sentences.append([labels, words])
                words = []
                labels = []
    docs.append(sentences)

number_test = int(len(docs) * 20 / 100)

with open("test.conll", "w", encoding="utf-8") as f:
    for doc in docs[0:number_test]:
        f.write("-DOCSTART- O\n\n")
        for sentence in doc:
            for word, label in zip(sentence[1], sentence[0]):
                f.write(word + " " + label + "\n")
        f.write("\n")

with open("train.conll", "w", encoding="utf-8") as f:
    for doc in docs[number_test:]:
        f.write("-DOCSTART- O\n\n")
        for sentence in doc:
            for word, label in zip(sentence[1], sentence[0]):
                f.write(word + " " + label + "\n")
        f.write("\n")
