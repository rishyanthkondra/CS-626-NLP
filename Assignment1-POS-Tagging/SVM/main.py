import svm
import nltk
import math

CORPUS_NAME = 'brown'
nltk.download(CORPUS_NAME)
nltk.download('universal_tagset')
CORPUS = nltk.corpus.brown
UNIVERSAL_TAGS = (
    "VERB",
    "NOUN",
    "PRON",
    "ADJ",
    "ADV",
    "ADP",
    "CONJ",
    "DET",
    "NUM",
    "PRT",
    "X",
    ".",
)


def test(model, test_set):
    tpos = {}
    fpos = {}
    fneg = {}
    totpos = 0
    tot = 0
    for sent in test_set:
        obtained_tags = model.pos_tag([word_tag[0] for word_tag in sent])
        true_tags = [word_tag[1] for word_tag in sent]
        for i in range(len(sent)):
            if true_tags[i] == obtained_tags[i]:
                tpos[true_tags[i]] = tpos.get(true_tags[i], 0) + 1
                totpos += 1
            else:
                fneg[true_tags[i]] = fneg.get(true_tags[i], 0) + 1
                fpos[obtained_tags[i]] = fpos.get(obtained_tags[i], 0) + 1
        tot += len(sent)
    precision = {}
    recall = {}
    f1 = {}
    accuracy = totpos/tot
    tags = list(tpos.keys())+list(fpos.keys())+list(fneg.keys())
    for tag in tags:
        precision[tag] = tpos.get(tag, 0)/(tpos.get(tag, 0)+fpos.get(tag, 0))
        recall[tag] = tpos.get(tag, 0)/(tpos.get(tag, 0)+fneg.get(tag, 0))
        f1[tag] = (2*precision[tag]*recall[tag])/(precision[tag]+recall[tag])
    print(precision)
    print(recall)
    print(f1)
    print(accuracy)


def validate(model, sents):
    ind = 0
    step = len(sents)//5
    esteps = len(sents)%5
    l = 0
    for i in range(5):
        r = l + step
        if esteps != 0:
            r += 1
            esteps -= 1
        train_set = sents[:l]+sents[r:]
        test_set = sents[l:r]
        model.train(train_set)
        test(model, test_set)
        print()
        break
        l = r


def _main():
    model = hmm.HMM(UNIVERSAL_TAGS)
    sents = CORPUS.tagged_sents(tagset='universal')
    validate(model, sents)

_main()