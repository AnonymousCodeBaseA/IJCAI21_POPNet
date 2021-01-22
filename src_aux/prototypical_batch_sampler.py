# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):


    def __init__(self, labels,Aux_labels, classes_per_it, num_samples, iterations,mode):

        print(classes_per_it,num_samples,iterations)
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.mode = mode
        if self.mode == 'train':
            self.child_label = Aux_labels[0]
            self.aux_label = Aux_labels[1]

            self.child_classes, _ = np.unique(self.child_label, return_counts=True)
            self.aux_classes, _ = np.unique(self.aux_label, return_counts=True)

            self.child_classes = torch.LongTensor(self.child_classes)
            self.aux_classes = torch.LongTensor(self.aux_classes)

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)


        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        # print(self.indexes.shape)
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        # c_idxs = torch.randperm(len(self.classes))[:cpi]
        # print(c_idxs.size())

        if self.mode == 'train':
            cpi_child = int(self.classes_per_it / 2)
            cpi_aux = self.classes_per_it - cpi_child

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            # c_idxs = torch.randperm(len(self.classes))[:cpi]
            
            if self.mode == 'train':

                c_idxs_child = torch.randperm(len(self.child_classes))[:cpi_child]
                tt = np.arange(len(self.child_classes),len(self.classes))
                np.random.shuffle(tt)
                c_idxs_aux = torch.from_numpy(tt).type(torch.LongTensor)[:cpi_aux]
                # c_idxs_aux = torch.randperm(len(self.aux_classes))[:cpi_aux]
                c_idxs = torch.cat((c_idxs_child,c_idxs_aux))
            else:
                c_idxs = torch.randperm(len(self.classes))[:cpi]

            # print(c_idxs_child)
            # print(c_idxs_aux)
            # print(c_idxs.size())
            #validate
            # ss = 0
            # for item in c_idxs:
            # 	if item in c_idxs_child:
            # 		ss+= 1

            # print(ss)

            # ss = 0
            # for item in c_idxs:
            # 	if item in c_idxs_aux:
            # 		ss+= 1

            # print(ss)

            # add here
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                # print(sample_idxs.size())
                batch[s] = self.indexes[label_idx][sample_idxs]
            # print(batch)
            batch = batch[torch.randperm(len(batch))]
            # print(batch)
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
