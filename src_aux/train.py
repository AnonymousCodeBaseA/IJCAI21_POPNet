# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from protonet import ProtoNet
from parser_util import get_parser
from flow_dataset import FlowDataset_Train
import flow_dataset
import random
from tqdm import tqdm
import shutil
import numpy as np
import torch
import os

import time
import csv

option = get_parser().parse_args()



# num_novel = 9 
num_known = 42

novel_catgs = []
known_catgs = []

novel_cal = []
knwon_cal = []

Novelty_detection =  option.large_scale
Cross_platform =  option.novelty_dec

if Cross_platform:
    num_novel = 9
else:
    num_novel = 0

def cal_ind_all(pre_y,y_true):
    y_pred = pre_y.flatten().tolist()
    y_true = y_true.flatten().tolist()
    for i in range(y_pred):
        true_label[y_true[i]] += 1
        pred_label[y_pred[i]] += 1


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)




def init_sampler(opt, labels,aux_labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    elif 'val' in mode:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
    								Aux_labels=aux_labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations,
                                    mode = mode)


def init_dataloader(opt, mode):
    
    dataset = FlowDataset_Train(mode)
    sampler = init_sampler(opt, dataset.label,dataset.aux_labels, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet().to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    global Novelty_detection
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    train_recall = []
    train_precision = []
    train_f1_score = []

    val_recall = []
    val_precision = []
    val_f1_score = []



    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tr_iter:
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)


            model_output = model(x.float())
            loss, acc , recall,precision, pred_y, label_y = loss_fn(model_output, target=y, n_support=opt.num_support_tr,mode = 'train')
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

            train_recall.append(recall)
            train_precision.append(precision)
            # train_f1_score.append(f1_score)

        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])

        avg_recall = np.mean(train_recall[-opt.iterations:])
        avg_precision = np.mean(train_precision[-opt.iterations:])
        # avg_f1_score = np.mean(train_f1_score[-opt.iterations:])

        print('Avg Train Loss: {}, Avg Train Acc: {}, Recall: {}, Precision: {}, f1_score: {}'.format(
            avg_loss, avg_acc, avg_recall, avg_precision, 2.0*avg_precision*avg_recall/(avg_precision+avg_recall)))

        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()

        true_label = np.zeros(50)
        pred_label = np.zeros(50)
        conf = np.zeros((42,42))

        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x.float())
            loss, acc , recall,precision, pred_y, label_y = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val,mode = 'test')

            y_pred = pred_y.flatten().tolist()
            y_true = label_y.flatten().tolist()
            

            for i in range(len(y_pred)):
                conf[y_pred[i]][y_true[i]] += 1
                if int(y_true[i]) == int(y_pred[i]):
                    pred_label[y_true[i]] += 1
                true_label[y_true[i]] += 1
        
                

            val_loss.append(loss.item())
            val_acc.append(acc.item())

            val_recall.append(recall)
            val_precision.append(precision)
            # val_f1_score.append(f1_score)
            # print(recall)
            # print(precision)
            # print(f1_score)
        


        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])

        avg_recall = np.mean(val_recall[-opt.iterations:])
        avg_precision = np.mean(val_precision[-opt.iterations:])
        # avg_f1_score = np.mean(val_f1_score[-opt.iterations:])
        # print(val_recall,val_precision,val_f1_score)
        save_files = '../cm/conf'

        np.savetxt(save_files + str(avg_acc) +'.txt',conf)

        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {} Recall: {}, Precision: {}, f1_score: {} {}'.format(
            avg_loss, avg_acc, avg_recall, avg_precision, 2.0*avg_precision*avg_recall/(avg_precision+avg_recall), postfix))

        

        #cal detection rate
        # for indx, item in enumerate(true_label):
        #     dr = pred_label[indx] * 1.00 /(true_label[indx]*1.00)
        #     if not np.isnan(dr):
        #         print(str(indx) +': ' + str(dr))

        if Novelty_detection is True or Cross_platform is True:
            tmp_dict = {}
            # for indx, item in enumerate(true_label):
            #     dr = pred_label[indx] * 1.00 /(true_label[indx]*1.00)
            #     if not np.isnan(dr):
            #         tmp_dict[indx] = dr

            novel_stat = []
            known_stat = []

            for item in tmp_dict.keys():
                if flow_dataset.val_catgs[item] in novel_catgs:
                    novel_stat.append((flow_dataset.val_catgs[item],tmp_dict[item]))

                if flow_dataset.val_catgs[item] in known_catgs:
                    known_stat.append(tmp_dict[item])

    print("Detection overall.")

    print('Knwon average ACC: {}'.format(float(np.array(known_stat).mean())))
    print('Novel ACC')
    for item in novel_stat:
        print(item[0] + ': '+str(item[1]))        


    if avg_acc >= best_acc:
        torch.save(model.state_dict(), best_model_path)
        best_acc = avg_acc
        best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):

    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    global num_novel, num_known
    val_recall = []
    val_precision = []
    val_f1_score = []

    for epoch in range(20):
        test_iter = iter(test_dataloader)

        true_label = np.zeros(50)
        pred_label = np.zeros(50)

        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x.float())
            loss, acc , recall,precision, pred_y, label_y = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val,mode = 'test')
            avg_acc.append(acc.item())

            y_pred = pred_y.flatten().tolist()
            y_true = label_y.flatten().tolist()

            conf = np.zeros((42,42))


            for i in range(len(y_pred)):
                conf[y_pred[i]][y_true[i]] += 1


                if int(y_true[i]) == int(y_pred[i]):
                    pred_label[y_true[i]] += 1
                true_label[y_true[i]] += 1

            

            val_recall.append(recall)
            val_precision.append(precision)
            # val_f1_score.append(f1_score)

    avg_recall = np.mean(val_recall[-opt.iterations:])
    avg_precision = np.mean(val_precision[-opt.iterations:])
    # avg_f1_score = np.mean(val_f1_score[-opt.iterations:])


    avg_acc = np.mean(avg_acc)

    print('Avg Val Acc: {} Recall: {}, Precision: {}, f1_score: {} '.format(
             avg_acc, avg_recall, avg_precision, 2.0*avg_precision*avg_recall/(avg_precision+avg_recall)))
    # print('Test Acc: {}'.format(avg_acc))
    # 
    #cal detection rate
    #
    for indx, item in enumerate(true_label):
        dr = pred_label[indx] * 1.00 /(true_label[indx]*1.00)
        if not np.isnan(dr):
            print(str(indx) +': ' + str(dr))

    if Novelty_detection is True or Cross_platform is True:
        tmp_dict = {}
        for indx, item in enumerate(true_label):
            dr = pred_label[indx] * 1.00 /(true_label[indx]*1.00)
            if not np.isnan(dr):
                print(str(indx) +': ' + str(dr))
                tmp_dict[indx] = dr

        novel_stat = []
        known_stat = []

        print(novel_catgs)
        for item in tmp_dict.keys():
            if flow_dataset.val_catgs[item] in novel_catgs:
                novel_stat.append((flow_dataset.val_catgs[item],tmp_dict[item]))

            if flow_dataset.val_catgs[item] in known_catgs:
                known_stat.append(tmp_dict[item])

        print("Detection overall.")

        with open('../record/' + 'Known_'  + str(num_known) +'_' + str(num_novel) + '.txt','a+') as f:

            f.write('\n')

            print('Knwon average ACC: {}'.format(float(np.array(known_stat).mean())))
            knwon_cal.append(float(np.array(known_stat).mean()))

            f.write(str(float(np.array(known_stat).mean())) + ' ')


            print('Novel ACC')
            tmp = []
            for item in novel_stat:
                print(item[0] + ': '+str(item[1]))
                tmp.append(item[1])
                f.write(str(item[1])+' ')

        if len(tmp) != num_novel:
            novel_cal.append(tmp[:num_novel])
        novel_cal.append(tmp)







    return avg_acc, avg_precision, avg_recall, 2.0*avg_precision*avg_recall/(avg_precision+avg_recall)


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)



def main():
    global num_novel, num_known
    #init datasets
    root_src = '../train'
    root_dst_val = '../val'
    root_dst_test = '../test'
    root_src_ustc = '../val-ustc'
    root_src_origin = '../val-origin'
    root_novel_backup = '../novel-backup'

    
    global Novelty_detection, Cross_platform

    
    if len(os.listdir(root_novel_backup)) != 0:
        dirs = os.listdir(root_novel_backup)
        for item in dirs:
            shutil.move(os.path.join(root_novel_backup,item),root_src)

    if len(os.listdir(root_dst_val)) != 0:
        shutil.rmtree(root_dst_val,True)
        shutil.rmtree(root_dst_test,True)



    catgs = os.listdir(root_src)

    catg_ind = np.arange(len(catgs))
    random.shuffle(catg_ind)

    if Novelty_detection is True:
        if len(os.listdir(root_src)) != 42:
            print('Training set is not sheer.')
            return

        if os.path.exists(root_dst_val) and len(os.listdir(root_dst_val)) != 0:
            print('Val is not empty.')
            return
        if not os.path.exists(root_dst_val):
            os.mkdir(root_dst_val)
            
        if not os.path.exists(root_dst_test):
            os.mkdir(root_dst_test)

    if Novelty_detection is True:
        #copy known class 
        for i in range(num_known):
            knwon_catg = catgs[catg_ind[i]]
            known_catgs.append(knwon_catg)

            shutil.copytree(os.path.join(root_src_origin,knwon_catg),os.path.join(root_dst_val,knwon_catg))
            shutil.copytree(os.path.join(root_src_origin,knwon_catg),os.path.join(root_dst_test,knwon_catg))

        #copy novel class
        catg_ind = catg_ind[num_known+num_novel:]
        for i in range(len(catg_ind)):

            novel_catg = catgs[catg_ind[i]]
            novel_catgs.append(novel_catg)

            shutil.move(os.path.join(root_src,novel_catg),root_novel_backup)

            shutil.copytree(os.path.join(root_src_origin,novel_catg),os.path.join(root_dst_val,novel_catg))
            shutil.copytree(os.path.join(root_src_origin,novel_catg),os.path.join(root_dst_test,novel_catg))


        print('Novelty detection starts.')


    if Cross_platform is True:


        if len(os.listdir(root_src)) != 42:
            print('Training set is not sheer.')
            return

        if os.path.exists(root_dst_val) and len(os.listdir(root_dst_val)) != 0:
            print('Val is not empty.')
            return
        if not os.path.exists(root_dst_val):
            os.mkdir(root_dst_val)
            
        if not os.path.exists(root_dst_test):
            os.mkdir(root_dst_test)

    if Cross_platform is True:
        #copy known class 
        for i in range(num_known):
            knwon_catg = catgs[catg_ind[i]]
            known_catgs.append(knwon_catg)

            shutil.copytree(os.path.join(root_src_origin,knwon_catg),os.path.join(root_dst_val,knwon_catg))
            shutil.copytree(os.path.join(root_src_origin,knwon_catg),os.path.join(root_dst_test,knwon_catg))

        #copy novel class
        catgs = os.listdir(root_src_ustc)

        catg_ind = np.arange(len(catgs))
        random.shuffle(catg_ind)


        for i in range(num_novel):

            novel_catg = catgs[catg_ind[i]]
            novel_catgs.append(novel_catg)

            shutil.move(os.path.join(root_src,novel_catg),root_novel_backup)

            shutil.copytree(os.path.join(root_src_ustc,novel_catg),os.path.join(root_dst_val,novel_catg))
            shutil.copytree(os.path.join(root_src_ustc,novel_catg),os.path.join(root_dst_test,novel_catg))


        print('Cross-Platform detection starts.')





    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')
    # trainval_dataloader = init_dataloader(options, 'trainval')
    test_dataloader = init_dataloader(options, 'test')

    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res


    model.load_state_dict(best_state)
    print('Testing with best model..')
    ac, pr, re, f1 = test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    save_files = options.save_file 

    with open(save_files,'a+') as f:
        writer = csv.writer(f)
        data = [(str(ac), str(pr), str(re), str(f1))]

        for i in data:
            writer.writerow(i)


if __name__ == '__main__':
    
    #main()

    times = 20

    
    for i in range(times):
        print("Evaluated times : {}".format(i))
        novel_catgs = []
        known_catgs = []

        np.random.seed(int(time.time()))
        main()



    known = float(np.array(knwon_cal).mean())

    # print(np.array(novel_cal))
    novelty = np.array(novel_cal).mean(0)

    print()
    print("Results Evaluated for {} times".format(times))
    print("Known class acc: {}".format(known))
    print("Novelty class acc: ")
    print(novelty)
