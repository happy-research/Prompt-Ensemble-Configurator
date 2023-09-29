import torch
from openprompt.data_utils import InputExample
import csv
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import random_split
from openprompt.prompts import ManualTemplate
import gc

#graphic card classes and label words
graphic_classes = [
    0,
    1,
    2,
    3,
    4
]
graphic_label_words={
    0:['NVIDIA GeForce GTX 1050', 'NVIDIA GeForce GTX 1050 Ti', 'NVIDIA GeForce GTX 1060', 'NVIDIA GeForce GTX 1070', 'NVIDIA GeForce 940MX'],
    1:['AMD Radeon R2', 'AMD Radeon R4', 'AMD Radeon R5 Graphics', 'AMD Radeon R7'],
    2:['Intel UHD Graphics 620', 'Intel Iris Plus Graphics 640', 'Intel HD Graphics 3000', 'Intel', 'Intel HD 620 graphics', 'Intel HD Graphics 500', 'Intel HD Graphics 520', 'Intel HD Graphics 620', 'Intel HD Graphics 400', 'Intel Celeron', 'Intel HD Graphics 505', 'Intel HD Graphics 5500', 'Intel HD Graphics', 'Intel?? HD Graphics 620 (up to 2.07 GB)', 'intel 620'],
    3:['Integrated', 'integrated intel hd graphics', 'integrated AMD Radeon R5 Graphics', 'Integrated Graphics', 'Integrated intel hd graphics'],
    4:['others'],
}
cpu_classes = [
    0,
    1,
    2,
    3
]
cpu_label_words = {
    0: ['4 GHz Intel Core i7'],
    1: ['3.8 GHz Intel Core i7', '3.8 GHz Core i7 Family', '3.5 GHz Intel Core i7', '3 GHz 8032', '3.5 GHz 8032', '3 GHz AMD A Series', '3.1 GHz Intel Core i5', '3.4 GHz Intel Core i5', '3.6 GHz AMD A Series', '3.5 GHz Intel Core i5', '3 GHz'],
    2: ['2.8 GHz Intel Core i7', '2.7 GHz Core i7 7500U', '2.7 GHz Core i7 2.7 GHz', '2.7 GHz Intel Core i7', '2.1 GHz Intel Core i7', '2.2 GHz Intel Core i5', '2.3 GHz Intel Core i5', '2.6 GHz Intel Core i5', '2.5 GHz Intel Core i5', '2.5 GHz Core i5 7200U', '2 GHz None', '2 GHz AMD A Series', '2.7 GHz Intel Core i3', '2.5 GHz Pentium', '2.5 GHz AMD A Series', '2.16 GHz Intel Celeron', '2.16 GHz Athlon 2650e', '2.7 GHz 8032', '2.48 GHz Intel Celeron', '2.4 GHz AMD A Series', '2 GHz Celeron D Processor 360', '2.4 GHz Intel Core i3', '2.3 GHz Intel Core i3', '2.4 GHz Core i3-540', '2.5 GHz Intel Core Duo', '2.2 GHz Intel Core i3', '2.7 GHz AMD A Series', '2.8 GHz 8032', '2.5 GHz Athlon 2650e', '2.9 GHz Intel Celeron', '2 GB'],
    3: ['1.5 GHz', '1.8 GHz 8032', '1.8 GHz AMD E Series', '1.7 GHz', '1.1 GHz Intel Celeron', '1.6 GHz Intel Celeron', '1.6 GHz Intel Core 2 Duo', '1.7 GHz Exynos 5000 Series', '1.6 GHz Celeron N3060', '1.6 GHz AMD E Series', '1.1 GHz Pentium', '1.6 GHz', '1.6 GHz Intel Mobile CPU', '1.6 GHz Celeron N3050', '1.8 GHz Intel Core i7', '1.6 GHz Intel Core i5'],
}
hard_classes = [
    0,
    1,
    2,
    3,
    4,
    5
]
hard_label_words = {
    0: ['2 TB HDD 5400 rpm'],
    1: ['1 TB', '1 TB HDD 7200 rpm', '1000 GB Mechanical Hard Drive', '1000 GB Hybrid Drive', '1 TB HDD 5400 rpm', '1024 GB Mechanical Hard Drive', '1 TB serial_ata', '1 TB mechanical_hard_drive', '1128 GB Hybrid'],
    2: ['500 GB HDD 5400 rpm', '500 GB mechanical_hard_drive', 'Solid State Drive, 512 GB', '512 GB SSD'],
    3: ['256 GB Flash Memory Solid State', '256 GB', '256.00 SSD', '256 GB SSD', '320 GB HDD 5400 rpm'],
    4: ['128 GB Flash Memory Solid State', '128 GB SSD'],
    5: ['others'],
}
ram_classes = [
    0,
    1,
    2,
    3,
    4,
    5,
    6
]
ram_label_words = {
    0: ['16 GB DDR4', '16 GB LPDDR3_SDRAM', '16 GB SDRAM', '16 GB DDR SDRAM'],
    1: ['12 GB', '12 GB DDR3', '12 GB DDR SDRAM'],
    2: ['8 GB SDRAM DDR3', '8 GB DDR3 SDRAM', '8 GB DDR4 2666MHz', '8 GB DDR4', '8 GB LPDDR3', '8 GB DDR4 SDRAM', '8 GB DDR4_SDRAM', '8 GB 2-in1 Media Card Reader, USB 3.1, Type-C', '8 GB DDR SDRAM', '8 GB SDRAM DDR4', '8 GB ddr4', '8 GB sdram', '8 GB SDRAM', '8 GB'],
    3: ['6 GB SDRAM', '6 GB', '6 GB SDRAM DDR4', '6 GB DDR SDRAM'],
    4: ['4 GB LPDDR3_SDRAM', '4 GB SDRAM DDR4', '4 GB ddr3_sdram', '4 GB DDR3', '4 GB SDRAM', '4 GB', '4 GB SDRAM DDR3', '4 GB DDR4', '4 GB DDR3 SDRAM', '4 GB DDR SDRAM'],
    5: ['2 GB SDRAM DDR3', '2 GB SDRAM', '2 GB DDR3L SDRAM', '2 GB DDR3 SDRAM'],
    6: ['others'],
}
scre_classes = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
]
scre_label_words = {
    0: ['19.5 inches'],
    1: ['17.3 inches'],
    2: ['15.6 inches'],
    3: ['14 inches'],
    4: ['13.5 inches', '13.3 inches'],
    5: ['12.5 inches', '12.3 inches'],
    6: ['11.6 inches'],
    7: ['10.1 inches'],
}

def copy_batch(batch,device):
    n_batch={}
    for key in batch.keys():
        ni=torch.tensor(batch[key],device=device)
        n_batch[key]=ni
    return n_batch
        

def read_data_csv(file,ratio):
    record=[]
    with open(file,newline='') as csvfile:
        read=csv.reader(csvfile)
        for item in read:
            record.append(item[1:])
    record=record[1:]
    for ind,sample in enumerate(record):
        sample.insert(0,ind)
        sample[2]=int(sample[2])#cpu
        sample[3]=int(sample[3])#graphic
        sample[4]=int(sample[4])#hardisk
        sample[5]=int(sample[5])#ram
        sample[6]=int(sample[6])#screen
    train_set, valid_set=random_split(record,
                 #[0.7,0.3],
                 ratio,
                 generator=torch.Generator().manual_seed(42))
    dataset={}
    train_dataset=[]
    valid_dataset=[]
    for item in train_set:
        train_dataset.append(InputExample(guid=item[0],text_a=item[1],label=item[2:]))
    for item in valid_set:
        valid_dataset.append(InputExample(guid=item[0],text_a=item[1],label=item[2:]))
    dataset['train']=train_dataset
    dataset['valid']=valid_dataset
    return dataset




class multiMask_MixTemplateModel(nn.Module):
    def __init__(self,
                plm:PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                WrapperClass,
                dataset,
                needdata,
                #classes,
                epoch,
                template_text,
                #shareTemplate,
                #label_words,
                device,
                cpu_classes,
                cpu_label_words,
                graphic_classes,
                graphic_label_words,
                hardisk_classes,
                hardisk_label_words,
                ram_classes,
                ram_label_words,
                screen_classes,
                screen_label_words,
                ):
        
        super().__init__()
        
        #self.promptTemplate = shareTemplate
        self.promptTemplate = MixedTemplate(
            model=plm,
            text = template_text,
            tokenizer = tokenizer,
        )

        # 5 verbalizer correpond to 5 attributes(cpu, graphic card, hard disk, ram, screen)
        self.cpu_promptVerbalizer = ManualVerbalizer(
            classes = cpu_classes,
            label_words = cpu_label_words,
            tokenizer = tokenizer,
            multi_token_handler="first",
        )
        self.cpu_promptVerbalizer.to(device)
        self.graphic_promptVerbalizer = ManualVerbalizer(
            classes = graphic_classes,
            label_words = graphic_label_words,
            tokenizer = tokenizer,
            multi_token_handler="first",
        )
        self.graphic_promptVerbalizer.to(device)
        self.hardisk_promptVerbalizer = ManualVerbalizer(
            classes = hardisk_classes,
            label_words = hardisk_label_words,
            tokenizer = tokenizer,
            multi_token_handler="first",
        )
        self.hardisk_promptVerbalizer.to(device)
        self.ram_promptVerbalizer = ManualVerbalizer(
            classes = ram_classes,
            label_words = ram_label_words,
            tokenizer = tokenizer,
            multi_token_handler="first",
        )
        self.ram_promptVerbalizer.to(device)
        self.screen_promptVerbalizer = ManualVerbalizer(
            classes = screen_classes,
            label_words = screen_label_words,
            tokenizer = tokenizer,
            multi_token_handler="first",
        )
        self.screen_promptVerbalizer.to(device)

        # Model backbone
        self.promptModel = PromptForClassification(
            template = self.promptTemplate,
            plm = plm,
            #verbalizer = self.promptVerbalizer,
            verbalizer = None,#rewrite model foward function and use the verbalizer outside
        )
        self.promptModel.to(device)

        #train_set, valid_set=random_split(dataset,
        #                                  [0.7,0.3],
        #                                  generator=torch.Generator().manual_seed(42))
        train_set=dataset['train']
        valid_set=dataset['valid']
        
        finetune_set=needdata['train']
        test_set=needdata['valid']

        self.train_data_loader = PromptDataLoader(
            dataset = train_set,
            tokenizer = tokenizer,
            template = self.promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=16,
            shuffle=True,
            #max_seq_length=800,
        )
        self.valid_data_loader = PromptDataLoader(
            dataset = valid_set,
            tokenizer = tokenizer,
            template = self.promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=16,
            #max_seq_length=800,
        )
        
        self.finetune_data_loader = PromptDataLoader(
            dataset = finetune_set,
            tokenizer = tokenizer,
            template = self.promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=16,
            shuffle=True,
            #max_seq_length=800,
        )
        self.test_data_loader = PromptDataLoader(
            dataset = test_set,
            tokenizer = tokenizer,
            template = self.promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=16,
            #max_seq_length=800,
        )

        self.cross_entropy  = nn.NLLLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in self.promptModel.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.promptModel.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # Using different optimizer for prompt parameters and model parameters
        optimizer_grouped_parameters2 = [
            {'params': [p for n,p in self.promptModel.template.named_parameters() if "raw_embedding" not in n]}
        ]
        self.optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-5)
        self.optimizer2 = AdamW(optimizer_grouped_parameters2, lr=1e-5)

        self.epoch=epoch

    def forward(self,batch):
        """outputs=self.promptModel(batch)"""
        #rewrite forward function
        
        outputs = self.promptModel.prompt_model(batch)
        
        #outputs = self.verbalizer.gather_outputs(outputs)
        outputs=outputs.logits#replace verbalizer's gather_outputs() function
        
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.promptModel.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.promptModel.extract_at_mask(outputs, batch)
        
        #use 5 verbalizers to replace original one
        #label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        outputs_at_mask=torch.transpose(outputs_at_mask,0,1)
        cpu_outputs_at_mask=outputs_at_mask[0]
        #print('hjh check----->',cpu_outputs_at_mask.shape)
        #cpu_outputs_at_mask=cpu_outputs_at_mask.view(cpu_outputs_at_mask.shape[0], cpu_outputs_at_mask.shape[2])
        cpu_label_words_logits = self.cpu_promptVerbalizer.process_outputs(cpu_outputs_at_mask, batch=batch)
        graphic_outputs_at_mask=outputs_at_mask[1]
        #graphic_outputs_at_mask=graphic_outputs_at_mask.view(graphic_outputs_at_mask.shape[0], graphic_outputs_at_mask.shape[2])
        graphic_label_words_logits = self.graphic_promptVerbalizer.process_outputs(graphic_outputs_at_mask, batch=batch)
        hardisk_outputs_at_mask=outputs_at_mask[2]
        #hardisk_outputs_at_mask=hardisk_outputs_at_mask.view(hardisk_outputs_at_mask.shape[0], hardisk_outputs_at_mask.shape[2])
        hardisk_label_words_logits = self.hardisk_promptVerbalizer.process_outputs(hardisk_outputs_at_mask, batch=batch)
        ram_outputs_at_mask=outputs_at_mask[3]
        #ram_outputs_at_mask=ram_outputs_at_mask.view(ram_outputs_at_mask.shape[0], ram_outputs_at_mask.shape[2])
        ram_label_words_logits = self.ram_promptVerbalizer.process_outputs(ram_outputs_at_mask, batch=batch)
        screen_outputs_at_mask=outputs_at_mask[4]
        #screen_outputs_at_mask=screen_outputs_at_mask.view(screen_outputs_at_mask.shape[0], screen_outputs_at_mask.shape[2])
        screen_label_words_logits = self.screen_promptVerbalizer.process_outputs(screen_outputs_at_mask, batch=batch)
        
        return cpu_label_words_logits, graphic_label_words_logits, hardisk_label_words_logits,\
                ram_label_words_logits, screen_label_words_logits

    def train(self):
        self.promptModel.train()

    def eval(self):
        self.promptModel.eval()
    
    def set_epoch(self,epoch):
        self.epoch=epoch






if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

    dataset=read_data_csv("/kaggle/input/reviewallmap/review_all_map.csv",[0.9,0.1])
    need_dataset=read_data_csv("/kaggle/input/needallmap/need_all_map.csv",[0.5,0.5])
    
    epoch=10
    
    #No.1 :
    template='{"soft": "unused unused unused"} {"placeholder":"text_a"} {"soft": "unused unused unused"} a computer with a {"mask"} {"mask"} {"mask"} {"mask"} {"mask"} {"soft": "unused"}'
    #No.2 :template='{"soft": "unused unused unused unused unused"} {"placeholder":"text_a"} {"mask"} {"mask"} {"mask"} {"mask"} {"mask"}'
    #No.3 try#template='{"soft": "unused unused unused unused unused unused unused"} {"placeholder":"text_a"} {"mask"} {"mask"} {"mask"} {"mask"} {"mask"}'
    #No.4 :template='{"soft": "unused unused unused unused"} {"placeholder":"text_a"} {"mask"} {"mask"} {"mask"} {"mask"} {"mask"} {"soft": "unused unused unused unused"}'
    #No.5 :template='{"soft": "unused unused unused unused"} {"placeholder":"text_a"} {"mask"} {"mask"} {"mask"} {"mask"} {"mask"} {"soft": "unused unused"}'
    #No.6 :unused unused …… unused unused
    #No.6 :template='{"soft": "unused unused"} {"placeholder":"text_a"} {"mask"} {"mask"} {"mask"} {"mask"} {"mask"} {"soft": "unused unused"}'
    #No.7 :template='{"soft": "unused unused"} {"placeholder":"text_a"} {"mask"} {"mask"} {"mask"} {"mask"} {"mask"} {"soft": "unused unused unused unused"}'
    
    
    model=multiMask_MixTemplateModel(plm,
                                   tokenizer,
                                   WrapperClass,
                                   dataset,
                                   need_dataset,
                                   #graphic_classes,
                                   epoch,
                                   #shareTemplate,
                                   template,
                                   #graphic_label_words,
                                   device,
                                   cpu_classes,
                                    cpu_label_words,
                                    graphic_classes,
                                    graphic_label_words,
                                    hard_classes,
                                    hard_label_words,
                                    ram_classes,
                                    ram_label_words,
                                    scre_classes,
                                    scre_label_words)
    

    
    #-----------------------Train-------------------------
    model.train()
    for i in range(model.epoch):
        count=0
        loss_rec=0
        for batch in model.train_data_loader:
            batch.to(device)
            
            labels=batch['label']
            label_trans=torch.transpose(batch['label'],0,1)
            cpu_labels=label_trans[0]
            graphic_labels=label_trans[1]
            hard_labels=label_trans[2]
            ram_labels=label_trans[3]
            scre_labels=label_trans[4]
            
            #share model
            cpu_logits, graphic_logits, hard_logits, ram_logits, scre_logits=model(batch)
            
            cpu_loss=model.cross_entropy(cpu_logits,cpu_labels)
            graphic_loss=model.cross_entropy(graphic_logits,graphic_labels)
            hard_loss=model.cross_entropy(hard_logits,hard_labels)
            ram_loss=model.cross_entropy(ram_logits,ram_labels)
            scre_loss=model.cross_entropy(scre_logits,scre_labels)
            
            shared_loss=cpu_loss+graphic_loss+hard_loss+ram_loss+scre_loss
            
            shared_loss.backward()
            model.optimizer1.step()
            model.optimizer1.zero_grad()
            model.optimizer2.step()
            model.optimizer2.zero_grad()
            
            count+=1
            loss_rec+=shared_loss
            
        gc.collect()
        torch.cuda.empty_cache()
        print('NO.',i,' epoch avg loss: ',loss_rec/count)
    
    
    
    #-----------------------Validate-------------------------
    model.eval()
    cpu_preds=[]
    cpu_labels=[]
    graphic_preds=[]
    graphic_labels=[]
    hard_preds=[]
    hard_labels=[]
    ram_preds=[]
    ram_labels=[]
    scre_preds=[]
    scre_labels=[]
    with torch.no_grad():
        #for step, inputs in enumerate(graphic_model.valid_data_loader):
        for step, inputs in enumerate(model.valid_data_loader):
            inputs.to(device)

            #share model
            cpu_logits, graphic_logits, hard_logits, ram_logits, scre_logits=model(inputs)

            #graphic_logits=graphic_model(inputs)
            #all_label=inputs['label']
            label_trans=torch.transpose(inputs['label'],0,1)
            cpu_label=label_trans[0]
            graphic_label=label_trans[1]
            hard_label=label_trans[2]
            ram_label=label_trans[3]
            scre_label=label_trans[4]

            cpu_labels.extend(cpu_label.cpu().tolist())
            cpu_preds.extend(torch.argmax(cpu_logits,dim=-1).cpu().tolist())
            graphic_labels.extend(graphic_label.cpu().tolist())
            graphic_preds.extend(torch.argmax(graphic_logits,dim=-1).cpu().tolist())
            hard_labels.extend(hard_label.cpu().tolist())
            hard_preds.extend(torch.argmax(hard_logits,dim=-1).cpu().tolist())
            ram_labels.extend(ram_label.cpu().tolist())
            ram_preds.extend(torch.argmax(ram_logits,dim=-1).cpu().tolist())
            scre_labels.extend(scre_label.cpu().tolist())
            scre_preds.extend(torch.argmax(scre_logits,dim=-1).cpu().tolist())
        
    cpu_acc=sum([int(i==j) for i,j in zip(cpu_preds, cpu_labels)])/len(cpu_preds)
    graphic_acc=sum([int(i==j) for i,j in zip(graphic_preds, graphic_labels)])/len(graphic_preds)
    hard_acc=sum([int(i==j) for i,j in zip(hard_preds, hard_labels)])/len(hard_preds)
    ram_acc=sum([int(i==j) for i,j in zip(ram_preds, ram_labels)])/len(ram_preds)
    scre_acc=sum([int(i==j) for i,j in zip(scre_preds, scre_labels)])/len(scre_preds)

    print("cpu accuracy is : ",cpu_acc)
    print("graphic card accuracy is : ",graphic_acc)
    print("hard disk accuracy is : ",hard_acc)
    print("ram accuracy is : ",ram_acc)
    print("scre accuracy is : ",scre_acc)

    
    #-----------------------Fine tune-------------------------
    model.train()
    for i in range(20):
        count=0
        loss_rec=0
        for batch in model.finetune_data_loader:
            batch.to(device)
            
            labels=batch['label']
            label_trans=torch.transpose(batch['label'],0,1)
            cpu_labels=label_trans[0]
            graphic_labels=label_trans[1]
            hard_labels=label_trans[2]
            ram_labels=label_trans[3]
            scre_labels=label_trans[4]
            
            #share model
            cpu_logits, graphic_logits, hard_logits, ram_logits, scre_logits=model(batch)
            
            cpu_loss=model.cross_entropy(cpu_logits,cpu_labels)
            graphic_loss=model.cross_entropy(graphic_logits,graphic_labels)
            hard_loss=model.cross_entropy(hard_logits,hard_labels)
            ram_loss=model.cross_entropy(ram_logits,ram_labels)
            scre_loss=model.cross_entropy(scre_logits,scre_labels)
            
            shared_loss=cpu_loss+graphic_loss+hard_loss+ram_loss+scre_loss
            
            shared_loss.backward()
            model.optimizer1.step()
            model.optimizer1.zero_grad()
            model.optimizer2.step()
            model.optimizer2.zero_grad()
            
            count+=1
            loss_rec+=shared_loss
            
        gc.collect()
        torch.cuda.empty_cache()
        print('NO.',i,' epoch avg loss: ',loss_rec/count)
        
        #-----------------------test-------------------------
        if(i==4 or i==9 or i==14 or i==19):
            with torch.no_grad():
                model.eval()
                cpu_preds=[]
                cpu_labels=[]
                cpu_all_pred=[]
                graphic_preds=[]
                graphic_labels=[]
                graphic_all_pred=[]
                hard_preds=[]
                hard_labels=[]
                hard_all_pred=[]
                ram_preds=[]
                ram_labels=[]
                ram_all_pred=[]
                scre_preds=[]
                scre_labels=[]
                scre_all_pred=[]
                
                for step, inputs in enumerate(model.test_data_loader):
                    inputs.to(device)
                    
                    cpu_logits, graphic_logits, hard_logits, ram_logits, scre_logits=model(inputs)

                    label_trans=torch.transpose(inputs['label'],0,1)
                    cpu_label=label_trans[0]
                    graphic_label=label_trans[1]
                    hard_label=label_trans[2]
                    ram_label=label_trans[3]
                    scre_label=label_trans[4]

                    cpu_labels.extend(cpu_label.cpu().tolist())
                    cpu_preds.extend(torch.argmax(cpu_logits,dim=-1).cpu().tolist())
                    cpu_all_pred.extend(cpu_logits.cpu().tolist())
                    graphic_labels.extend(graphic_label.cpu().tolist())
                    graphic_preds.extend(torch.argmax(graphic_logits,dim=-1).cpu().tolist())
                    graphic_all_pred.extend(graphic_logits.cpu().tolist())
                    hard_labels.extend(hard_label.cpu().tolist())
                    hard_preds.extend(torch.argmax(hard_logits,dim=-1).cpu().tolist())
                    hard_all_pred.extend(hard_logits.cpu().tolist())
                    ram_labels.extend(ram_label.cpu().tolist())
                    ram_preds.extend(torch.argmax(ram_logits,dim=-1).cpu().tolist())
                    ram_all_pred.extend(ram_logits.cpu().tolist())
                    scre_labels.extend(scre_label.cpu().tolist())
                    scre_preds.extend(torch.argmax(scre_logits,dim=-1).cpu().tolist())
                    scre_all_pred.extend(scre_logits.cpu().tolist())
                    
                    
                    
                cpu_acc=sum([int(i==j) for i,j in zip(cpu_preds, cpu_labels)])/len(cpu_preds)
                graphic_acc=sum([int(i==j) for i,j in zip(graphic_preds, graphic_labels)])/len(graphic_preds)
                hard_acc=sum([int(i==j) for i,j in zip(hard_preds, hard_labels)])/len(hard_preds)
                ram_acc=sum([int(i==j) for i,j in zip(ram_preds, ram_labels)])/len(ram_preds)
                scre_acc=sum([int(i==j) for i,j in zip(scre_preds, scre_labels)])/len(scre_preds)

            save_path="/kaggle/working/mixprompt_NoFT_all_epoch_"+str(i+1)+"_test_res.csv"
            n=len(cpu_labels)
            record=[]
            for j in range(0,n):
                tmp={"index":j, "cpu_label":cpu_labels[j], "cpu_prediction":cpu_preds[j], "cpu_all_pred": cpu_all_pred[j],\
                                "graphic_label":graphic_labels[j], "graphic_prediction":graphic_preds[j], "graphic_all_pred": graphic_all_pred[j],\
                                "hard_label":hard_labels[j], "hard_prediction":hard_preds[j], "hard_all_pred": hard_all_pred[j],\
                                "ram_label":ram_labels[j], "ram_prediction":ram_preds[j], "ram_all_pred": ram_all_pred[j],\
                                "screen_label":scre_labels[j], "screen_prediction":scre_preds[j], "screen_all_pred": scre_all_pred[j]}
                record.append(tmp)

            with open(save_path, 'w', newline='') as csvfile:
                fieldnames = ['index', 'cpu_label','cpu_prediction','cpu_all_pred',\
                             'graphic_label','graphic_prediction','graphic_all_pred',\
                             'hard_label','hard_prediction','hard_all_pred',\
                             'ram_label','ram_prediction','ram_all_pred',\
                             'screen_label','screen_prediction','screen_all_pred']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(record)
            print(save_path)
            print(i," epoch test cpu accuracy is : ",cpu_acc)
            print(i," epoch test graphic card accuracy is : ",graphic_acc)
            print(i," epoch test hard disk accuracy is : ",hard_acc)
            print(i," epoch test ram accuracy is : ",ram_acc)
            print(i," epoch test scre accuracy is : ",scre_acc)