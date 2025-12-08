import yaml
from glob import glob
from .config import Config as cfg

class setUtil:
    
    def img_set_util():
        
        # get img path
        train_img_list = glob(cfg.train_img_list)
        valid_img_list = glob(cfg.valid_img_list)
        
        print(f'Train Data: {len(train_img_list)}, Test Data: {len(valid_img_list)}')
        
        # write txt file
        with open(cfg.train_txt, 'w') as f:
            f.write('\n'.join(train_img_list) + '\n')
            
        with open(cfg.valid_txt, 'w') as f:
            f.write('\n'.join(valid_img_list) + '\n')
            
    
    def yaml_upload(object_list):
        with open('./dataUtil/model_dataset/data.yaml', 'r') as f:
            data = yaml.load(f,Loader=yaml.FullLoader)        
            print(data)  

        data['nc'] = len(object_list)
        data['names'] = object_list
        

        with open('./dataUtil/model_dataset/data.yaml', 'w') as f:
            yaml.dump(data, f)
            print(data)