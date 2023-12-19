import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import tqdm


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, name, pid2name, encoder, N, K, Q, root, ispubmed=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        pid2name_path = os.path.join(root, pid2name + ".json")
        if not os.path.exists(path) or not os.path.exists(pid2name_path):
            print("[ERROR] Data file does not exist!")
            assert 0
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.ispubmed = ispubmed
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask

    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask

    def __additem__(self, d, word, pos1, pos2, mask):

        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        relation_set = {'word': [], 'mask': []}
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        for i, class_name in enumerate(target_classes):
            if self.ispubmed:
                if class_name in self.pid2name.keys():
                    name, _ = self.pid2name[class_name]
                    rel_text, rel_text_mask = self.__getname__(name)
                else:
                    rel_text, rel_text_mask = self.__getname__(class_name)
            else:
                rel_text, rel_text_mask = self.__getrel__(self.pid2name[class_name])
            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)

            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [i] * self.Q

        return support_set, query_set, query_label, relation_set

    def __len__(self):
        return 1000000000

def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_relation = {'word': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels, relation_sets = zip(*data)

    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]
        batch_label += query_labels[i]

    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, batch_relation


def get_loader(name, pid2name, encoder, N, K, Q, batch_size,
               num_workers=8, collate_fn=collate_fn, ispubmed=False, root='./data'):
    dataset = FewRelDataset(name, pid2name, encoder, N, K, Q, root, ispubmed)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


class FewRelTestDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, pid2name, encoder, N, K, Q, root, ispubmed=False):
        self.root = root
        path = os.path.join(root, name + ".json") # json file path
        

        pid2name_path = os.path.join(root, pid2name + ".json")
        
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        #self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder
        self.ispubmed = ispubmed

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        
    
    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask

    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask
    
    
    def __getitem__(self, index):
        #target_classes = random.sample(self.classes, self.N)
        relation_set = {'word': [], 'mask': []}
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        
        
        count = 0
        data = self.json_data[index]
        support_set_my = data['meta_train']
        rel_set = data['relation']
        
        for idx, j in enumerate(support_set_my):
            ##TODO
            #relation
            class_name = rel_set[idx]
            if self.ispubmed:
                if class_name in self.pid2name.keys():
                    name, _ = self.pid2name[class_name]
                    rel_text, rel_text_mask = self.__getname__(name)
                else:
                    rel_text, rel_text_mask = self.__getname__(class_name)
            else:
                rel_text, rel_text_mask = self.__getrel__(self.pid2name[class_name])

            # rel_text, rel_text_mask = self.__getrel__(self.pid2name[rel_set[idx]])
            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)

            
            for i in j:
                word, pos1, pos2, mask = self.__getraw__(i)
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                #support.append(word)
                self.__additem__(support_set, word, pos1, pos2, mask)
                 
        
            query_set_my = data['meta_test']
        #for j in query_set:
        
            word, pos1, pos2, mask = self.__getraw__(query_set_my)
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
        #support.append(word)
            self.__additem__(query_set, word, pos1, pos2, mask)

            query_label += [idx] * self.Q
            
        return support_set, query_set, query_label, relation_set  #separate support and query -- no pair
    
    def __len__(self):
        return 1000000000


def collate_fn22(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_relation = {'word': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels, relation_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        #TODO
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]
        #TODO
        batch_label += query_labels[i]
    
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    #TODO
    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)
    #TODO
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, batch_relation

def get_loader22(name, pid2name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn22, ispubmed=False, root='./data'):
    dataset = FewRelTestDataset(name, pid2name, encoder, N, K, Q, root, ispubmed)
    ##TODO
    #import pdb
    #pdb.set_trace()
    
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


'''LPD'''   
class FewRelDatasetLpd(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, args, name, encoder, N, K, Q, root, label_mask_prob=0.0):
        self.name = name
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))

        self.semeval = False
        self.pubmed = False
        if 'semeval' in path:
            self.semeval = True
        if 'pubmed' in path:
            self.pubmed = True
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        # self.debug = args.debug
        self.label_mask_prob = label_mask_prob
        self.encoder = encoder
        pid2name  = json.load(open('./data/pid2name.json'))
        self.label2desc = {}
        for key in pid2name.keys():
            self.label2desc[key] = [pid2name[key][0], pid2name[key][1]]
        del pid2name

    def __additem__(self, d, item, classname, add_relation_desc):

        if add_relation_desc and random.random() > self.label_mask_prob:
            if self.semeval:
               
                index = list(classname).index('(')
                label_des =list(classname)[:index]
                new = ""
                for x in label_des:
                    new += x 
                label_des = new
                label_des = label_des.split('-')
            elif self.pubmed:
                label_des = classname.split('_')
            else:

                label_des = self.label2desc[classname][1] 
                label_des = label_des.split()
  
            token_ls = label_des + [':'] + item["tokens"]
            head_pos = []
            tail_pos = []


            for i in item['h'][2][0]:
                head_pos.append(i+len(label_des)+1)
            for i in item['t'][2][0]:
                tail_pos.append(i+len(label_des)+1)
        else:
            token_ls = item["tokens"]
            head_pos = item['h'][2][0]
            tail_pos = item['t'][2][0]

        word, pos1, pos2, mask = self.encoder.tokenize(token_ls,
            head_pos,
            tail_pos)
        
        word = torch.tensor(word).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        mask = torch.tensor(mask).long()
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

        return token_ls

    def __getitem__(self, index):
        target_classes_id = random.sample(range(len(self.classes)), self.N)
        target_classes = [self.classes[class_id] for class_id in target_classes_id]
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_label = []
        mlp_label = []
    
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))
     
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                
                if count < self.K:
                    show = self.__additem__(support_set, self.json_data[class_name][j], class_name, add_relation_desc=True)
                    # if self.debug and self.label_mask_prob != 0:
                    #     print("Support")
                    #     print(show)
                else:
                    show = self.__additem__(query_set, self.json_data[class_name][j], class_name, add_relation_desc=False)
                    # if self.debug and self.label_mask_prob != 0:
                    #     print("Query")
                    #     print(show)
                count += 1

            query_label += [i] * self.Q
            mlp_label += [target_classes_id[i]] * self.K

        return support_set, query_set, query_label, mlp_label
    
    def __len__(self):
        return 1000000000
    
def collate_fn_lpd(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    batch_mlp_label = []
    support_sets, query_sets, query_labels, mlp_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
        batch_mlp_label += mlp_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    batch_mlp_label =  torch.tensor(batch_mlp_label)
    return batch_support, batch_query, batch_label, batch_mlp_label

def get_loader_lpd(args, name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_lpd,  root='./data', label_mask_prob=1.0):

    dataset = FewRelDatasetLpd(args, name, encoder, N, K, Q,  root, label_mask_prob)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

'''LPD test'''
class FewRelTestDatasetLpd(data.Dataset):
    def __init__(self, name, encoder, root, label_mask_prob=0.0):
        self.name = name
        self.root = root
        path = os.path.join(root, self.name + ".json") 
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.pubmed = False
        if 'pubmed' in path:
            self.pubmed = True
   
        self.encoder = encoder
        pid2name  = json.load(open('./data/pid2name.json'))
        self.label2desc = {}
        for key in pid2name.keys():
            self.label2desc[key] = [pid2name[key][0], pid2name[key][1]]
        del pid2name
        self.data = []
        self.label_mask_prob =label_mask_prob
        self.__prepare_data__()

    def __prepare_data__(self): 
        content = self.json_data   
        sup_count = 0
        query_count = 0 
        for episode in tqdm(content):
            N = len(episode['meta_train'])
            K = len(episode['meta_train'][0])
            Q = 1
            support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
            query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
            for K_relation, support_label in zip(episode['meta_train'], episode['relation']):
                for single_instance in K_relation:
                    show = self.__additem__(support, single_instance, support_label, True)
                    if sup_count  < 50:
                        print('Support')
                        sup_count+=1
                        print(show)
         
            show = self.__additem__(query, episode['meta_test'], None, False)
            if query_count < 10:
                print("Query:")
                query_count+=1
                print(show)
            self.data.append([support, query])
        
    def __additem__(self, d, item, classname, add_relation_desc):
        if add_relation_desc and random.random() >= self.label_mask_prob:
            if self.pubmed:
                label_des = classname.split('_')
            else:

                label_des = self.label2desc[classname][1] 
                label_des = label_des.split()
            token_ls = label_des + [':'] + item["tokens"]
            head_pos = []
            tail_pos = []
            for i in item['h'][2][0]:
                head_pos.append(i+len(label_des)+1)
            for i in item['t'][2][0]:
                tail_pos.append(i+len(label_des)+1)
        else:
            token_ls = item["tokens"]
            head_pos = item['h'][2][0]
            tail_pos = item['t'][2][0]

        word, pos1, pos2, mask = self.encoder.tokenize(token_ls,
            head_pos,
            tail_pos)
        
        word = torch.tensor(word).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        mask = torch.tensor(mask).long()
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

        return token_ls

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def collate_fn_test_lpd(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

    support_sets, query_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
      
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
  
    return batch_support, batch_query

def get_loader_test_lpd(name, encoder, batch_size, num_workers=8, collate_fn=collate_fn_lpd, root='./data'):
    dataset = FewRelTestDatasetLpd(name, encoder, root)
    ##TODO
    #import pdb
    #pdb.set_trace()
    
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)    


'''BERTPAIR'''  
class FewRelDatasetPair(data.Dataset):
    """
    FewRel Pair Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            print(path)
            print(root)
            print(name)
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support = []
        query = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))
        #5
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False) ##generate random number from [0, #class_number]  K: support instances   Q: query instances 1
            count = 0
            for j in indices:
                word  = self.__getraw__(
                        self.json_data[class_name][j])
                if count < self.K:
                    support.append(word)
                else:
                    query.append(word)
                count += 1
        
            query_label += [i] * self.Q
        
        #two list: 'support' 'query' contain the support word and the query word
        
        ##TODO
        #import pdb
        #pdb.set_trace()
        
        #*****# when Q_na = 0, next code is not implemented.
        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word = self.__getraw__(
                    self.json_data[cur_class][index])
            query.append(word)
        query_label += [self.N] * Q_na
        #*****# when Q_na = 0, above code is not implemented.
        
        for word_query in query:
            for word_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])     
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label

def get_loader_pair(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_pair, na_rate=0, root='./data', encoder_name='bert'):
    dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)  

'''test'''
class FewRelTestPair(data.Dataset):
    """
    FewRel Pair Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            print(path)
            print(root)
            print(name)
            assert(0)
        self.json_data = json.load(open(path)) ##list type
        #self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        #target_classes = random.sample(self.classes, self.N)
        support = []
        query = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        #na_classes = list(filter(lambda x: x not in target_classes,  
        #    self.classes))
        ##
        
        #import pdb
        #pdb.set_trace()
        
        count = 0
        data = self.json_data[index]
        #for i in range(index, index + self.N):
        #support
        #data = self.json_data[i]
        support_set = data['meta_train']
        # for j in support_set:
        #     word = self.__getraw__(j[0])
        #     support.append(word)
              
        #     query_set = data['meta_test']
        #     word = self.__getraw__(query_set)
        #     query.append(word)

        for j in support_set:
            for k in j:
                word = self.__getraw__(k)
                support.append(word)
                
            query_set = data['meta_test']
            word = self.__getraw__(query_set)
            query.append(word)
        
        # print(len(support))
        # print(len(query))
       
        
        for word_query in query:
            for word_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])     
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set#, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn_pair2(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    #batch_label = []
    fusion_sets = data#zip(*data)
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        #batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    #batch_label = torch.tensor(batch_label)
    # print(batch_set['word'].size())
    return batch_set#, batch_label

def get_loader_pair2(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_pair2, na_rate=0, root='./data', encoder_name='bert'):
    dataset = FewRelTestPair(name, encoder, N, K, Q, na_rate, root, encoder_name)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


class FewRelDatasetReg(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos, mask = self.encoder.tokenize_reg(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos, mask 

    def __additem__(self, d, word, pos, mask, rel):
        d['word'].append(word)
        d['pos'].append(pos)
        d['mask'].append(mask)
        d['rel'].append(rel)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos': [], 'mask': [], 'rel': [] }
        query_set = {'word': [], 'pos': [], 'mask': [], 'rel': [] }
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos, mask = self.__getraw__(
                        self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos = torch.tensor(pos).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos, mask, class_name)
                else:
                    self.__additem__(query_set, word, pos, mask, class_name)
                count += 1

            query_label += [i] * self.Q

        # NA
        '''
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
        query_label += [self.N] * Q_na
        '''

        return support_set, query_set, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn_reg(data):
    batch_support = {'word': [], 'pos': [], 'mask': [], 'rel': []}
    batch_query = {'word': [], 'pos': [], 'mask': [], 'rel': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        if k == 'rel':
            continue
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        if k == 'rel':
            continue
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label

def get_loader_reg(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_reg, na_rate=0, root='./data'):
    dataset = FewRelDatasetReg(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


'''REGRAB test'''
## FewRelTestReg
############
class FewRelTestDatasetReg(data.Dataset):
    """
    FewRel Dataset for REGRAB
    """

    def __init__(self, name, encoder, N, K, Q, root):
        self.root = root
        path = os.path.join(root, name + ".json")  ## --test = name =  'test_wiki5-1'
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))  ## ./data/test_wiki5-1.json

        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder


    def __getraw__(self, item):
        word, pos, mask = self.encoder.tokenize_reg(list(item.items())[0][1],
                                                    item['h'][2][0],
                                                    item['t'][2][0])
        return word, pos, mask



    def __additem__(self, d, word, pos, mask, rel):
        d['word'].append(word)
        d['pos'].append(pos)
        d['mask'].append(mask)
        d['rel'].append(rel)



    def __getitem__(self, index):
        # target_classes = self.classes
        support_set = {'word': [], 'pos': [], 'mask': [], 'rel': []}
        query_set = {'word': [], 'pos': [], 'mask': [], 'rel': []}
        query_label = []

        data = self.json_data[index]
        support_set_my = data['meta_train']
        query_set_my = data['meta_test']

        class_name = data['relation']



        for i in support_set_my:   
            for j in i:
                word, pos, mask = self.__getraw__(j)   
                word = torch.tensor(word).long()
                pos = torch.tensor(pos).long()
                mask = torch.tensor(mask).long()
                self.__additem__(support_set, word, pos, mask, class_name)

            word, pos, mask = self.__getraw__(query_set_my)
            word = torch.tensor(word).long()
            pos = torch.tensor(pos).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos, mask, class_name)

        query_label += [0] * self.Q

        return support_set, query_set, query_label


    def __len__(self):
        return 1000000000


def get_loader_reg2(name, encoder, N, K, Q, batch_size,
                        num_workers=8, collate_fn=collate_fn_reg, root='./data'):
    dataset = FewRelTestDatasetReg(name, encoder, N, K, Q, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)




def get_loader_reg22():
    pass

