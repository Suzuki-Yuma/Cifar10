import numpy as np
import chainer
import argparse
xp = np
if chainer.cuda.available:
    import cupy
    xp = cupy


class ConvBn(chainer.Chain):
    def __init__(self, out_size, ksize, stride, pad):
        super(ConvBn, self).__init__()
        with self.init_scope():
            self.conv = chainer.links.Convolution2D(
                None, out_size, ksize, stride=stride, pad=pad, nobias=True)
            self.bn = chainer.links.BatchNormalization(out_size)

    def __call__(self, x):
        return chainer.functions.relu(self.bn(self.conv(x)))


class CNN(chainer.Chain):
    def __init__(self, class_num):
        super(CNN, self).__init__()
        with self.init_scope():
            self.block1 = ConvBn(64, 3, 1, 1)
            self.block2 = ConvBn(64, 3, 1, 1)
            self.pool1 = ConvBn(128, 3, 2, 0)

            self.block3 = ConvBn(128, 3, 1, 1)
            self.block4 = ConvBn(128, 3, 1, 1)
            self.pool2 = ConvBn(256, 3, 2, 0)

            self.block5 = ConvBn(256, 3, 1, 1)
            self.block6 = ConvBn(256, 3, 1, 1)
            self.block7 = ConvBn(256, 3, 1, 1)
            self.pool3 = ConvBn(512, 3, 2, 0)

            self.fc1 = chainer.links.Linear(
                None, 512, initialW=chainer.initializers.HeNormal(1e-3))
            self.fc_ln = chainer.links.LayerNormalization(512)
            self.fc2 = chainer.links.Linear(
                None, 10, initialW=chainer.initializers.HeNormal(1e-3))

    def __call__(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = chainer.functions.dropout(self.pool1(h), ratio=0.2)

        h = self.block3(h)
        h = self.block4(h)
        h = chainer.functions.dropout(self.pool2(h), ratio=0.2)

        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = chainer.functions.dropout(self.pool3(h), ratio=0.2)

        h = chainer.functions.relu(self.fc_ln(self.fc1(h)))
        h = chainer.functions.dropout(h, ratio=0.5)
        return self.fc2(h)


def BC_preprocess(train, test):
    images = ()
    labels = ()
    print("Processing for BC learning...", len(train))
    for i in range(len(train)):
        image1, label1 = train[np.random.randint(0, len(train))]
        image2, label2 = train[np.random.randint(0, len(train))]
        label1 = np.eye(10)[label1].astype(np.float32)
        label2 = np.eye(10)[label2].astype(np.float32)
        r = np.random.rand()
        images += (image1 * r + image2 * (1 - r),)
        labels += (label1 * r + label2 * (1 - r),)
    images2 = ()
    labels2 = ()
    for i in range(len(test)):
        image, label = test[i]
        label = np.eye(10)[label].astype(np.float32)
        images2 += (image, )
        labels2 += (label, )
    return chainer.datasets.TupleDataset(images, labels), chainer.datasets.TupleDataset(images2, labels2)


def KL_loss(y, t):
    ent = chainer.functions.sum(t[t != 0.] *
                                  chainer.functions.log(t[t != 0.]))
    cr_ent = chainer.functions.sum(t * chainer.functions.log_softmax(y))
    return (ent - cr_ent) / y.shape[0]


def cos_sim(y, t):
    y_ = chainer.Variable(xp.eye(10).astype(xp.float32))[
        chainer.cuda.to_cpu(chainer.functions.argmax(y, axis=1).data)]
    return chainer.functions.mean(chainer.functions.sum(y_ * t, axis=1) / chainer.functions.sqrt(chainer.functions.batch_l2_norm_squared(t) * chainer.functions.batch_l2_norm_squared(y_)))


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    train, test = chainer.datasets.get_cifar10()
    train, test = BC_preprocess(train, test)

    print(len(train), len(test))

    model = chainer.links.Classifier(CNN(10), lossfun=KL_loss, accfun=cos_sim)

    if args.gpu >= 0 and chainer.cuda.available:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    opt = chainer.optimizers.Adam()
    opt.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(
        train_iter, opt, device=args.gpu)
    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(chainer.training.extensions.Evaluator(
        test_iter, model, device=args.gpu))
    trainer.extend(chainer.training.extensions.dump_graph('main/loss'))
    trainer.extend(chainer.training.extensions.snapshot(),
                   trigger=(args.epoch, 'epoch'))
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(chainer.training.extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
