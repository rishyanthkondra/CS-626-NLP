class HMM:
    def __init__(self, tagset):
        self.k=0.0001
        self.tagset = tagset

    def train(self, sents):
        self.emit = {}
        self.trans = {}
        self.freq = {}
        for sent in sents:
            prev_tag='^'
            for ind, word in enumerate(sent):
                word_tag = (word[0].lower(), word[1])
                bigram = (prev_tag, word[1])
                self.emit[word_tag] = self.emit.get(word_tag, 0) + 1
                self.freq[word[1]] = self.freq.get(word[1], 0) + 1
                if ind != 0:
                    self.trans[bigram] = self.trans.get(bigram, 0) + 1
                    prev_tag = word[1]
        for tag in self.tagset:
            self.freq[tag] = self.freq.get(tag, 0)


    def laplace_trans(self, bigram):
        return (self.trans.get(bigram, 0)+self.k)/(self.freq[bigram[0]]+len(self.tagset)*self.k)


    def laplace_emit(self, word_tag):
        return (self.emit.get(word_tag, 0)+self.k)/(self.freq[word_tag[1]]+len(self.tagset)*self.k)


    def pos_tag(self, sent):
        dp = {}
        back = {}
        for ind, word in enumerate(sent):
            word = word.lower()
            if ind == 0:
                for tag in self.tagset:
                    word_tag = (word, tag)
                    dp[(ind, tag)] = self.laplace_emit(word_tag)
                continue
            for tag in self.tagset:
                dp[(ind, tag)] = -1
                for prev_tag in self.tagset:
                    bigram = (prev_tag, tag)
                    word_tag = (word, tag)
                    prob = dp[(ind-1, prev_tag)]*self.laplace_trans(bigram)*self.laplace_emit(word_tag)
                    if dp[(ind, tag)] < prob:
                        dp[(ind, tag)] = prob 
                        back[(ind, tag)] = prev_tag
        tags = []
        max_prob = -1
        for tag in self.tagset:
            if max_prob < dp[len(sent)-1, tag]:
                max_prob = dp[len(sent)-1, tag]
                tags = [tag]

        for ind in range(len(sent)-1, 0, -1):
            tags.append(back[(ind, tags[-1])])
        tags.reverse()
        return tags
