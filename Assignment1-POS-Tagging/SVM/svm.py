import numpy


class SVM:
    def __init__(self, tagset):
        self.tag2num = {}
        self.num2tag = {}
        for i in range(len(tagset)):
            self.tag2num[tagset[i]] = i
            self.num2tag[i] = tagset[i]
        return


    def encode(self, word):
        return list(range(10))


    def train(self, sents):
        X = []
        y = []
        for sent in sents:
            for word_tag in sent:
                X.append(self.encode(word_tag[0]))
                y.append(self.tag2num[word_tag[1]])
        X = numpy.array(X)
        y = numpy.array(y)
        self.W = numpy.zeros((X.shape[1], len(self.tag2num)))

        iters = 50
        batch_size = 100
        lmbda = 1e-4
        lr = 1e-2
        losses = []
        for i in range(iters):
            batch_inds = numpy.random.choice(X.shape[0], batch_size)
            X_batch = X[batch_inds,:]
            y_batch = y[batch_inds]
            s = X_batch.dot(self.W)

            #loss
            correct_s = s[list(range(batch_size)), y_batch]
            s = 1 + s - correct_s.reshape(-1, 1)
            s[list(range(batch_size)), y_batch] = 0
            loss = np.sum(np.fmax(s, 0))/num_train
            loss += lmbda * np.sum(W*W)

            #gradient
            X_mask = s
            X_mask[X_mask > 0] = 1
            X_mask[list(range(batch_size)), y] = -np.sum(X_mask, axis=1)
            grad = X_batch.T.dot(X_mask)
            grad = (grad/num_train) + 2*reg*W

            #update W
            W -= lr * grad
            if i%10==0:
                print("Iteration: {}, Loss: {}".format(i, loss))

        return


    def pos_tags(self, sents):
        out = []
        for sent in sents:
            tags = []
            for word in sent:
                vec = self.encode(word)
                y = numpy.argmax(vec.dot(self.W), axis=1)
                tags.append(self.num2tag[y])
            out.append(tags)
        return out

