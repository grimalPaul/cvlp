import random

# train_loaders.append(Dataloader)
# train_loader = MultitaskLoader(
#         train_loaders,
#         sampling=args.multitask_sampling,
#         verbose=gpu==0)

class MultitaskLoader(object):
    def __init__(self, loaders, shuffle=True, sampling='roundrobin', n_batches=None, verbose=True):
        self.loaders = loaders
        self.verbose = verbose

        self.task2len = {loader.task: len(loader) for loader in self.loaders}
        if self.verbose:
            print('Task2len:', self.task2len)
        self.task2loader = {loader.task: loader for loader in self.loaders}

        self.shuffle = shuffle
        self.sampling = sampling
        self.epoch_tasks = None
        self.n_batches = n_batches
        self.set_epoch(0)

    def __iter__(self):
        self.task2iter = {loader.task: iter(loader) for loader in self.loaders}
        return self

    def set_epoch(self, epoch):
        for loader in self.loaders:
            loader.sampler.set_epoch(epoch)

        if self.sampling == 'roundrobin':
            epoch_tasks = []
            for task, loader in self.task2loader.items():
                n_batches = len(loader)
                epoch_tasks.extend([task]*n_batches)
        elif self.sampling == 'balanced':
            if self.n_batches is None:
                n_batches = sum(self.task2len.values()) // len(self.loaders)
            else:
                n_batches = self.n_batches
            if self.verbose:
                print('# batches:', n_batches)
            epoch_tasks = []
            for task, loader in self.task2loader.items():
                epoch_tasks.extend([task]*n_batches)

        if self.shuffle:
            random.Random(epoch).shuffle(epoch_tasks)
        self.epoch_tasks = epoch_tasks
        if self.verbose:
            print('# epoch_tasks:', len(self.epoch_tasks))

    def __next__(self):
        if len(self.epoch_tasks) > 0:
            task = self.epoch_tasks.pop()
            loader_iter = self.task2iter[task]
            return next(loader_iter)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.epoch_tasks)

def get_multitask_loader(loaders :list, verbose):
    return MultitaskLoader(loaders, verbose=verbose)

def get_val_loader(loaders:list):
    val_loader = {}
    for loader in loaders:
        val_loader[loader.task] = loader
    return val_loader

if __name__ == '__main__':
    pass