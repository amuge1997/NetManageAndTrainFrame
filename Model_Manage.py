import json,os,shutil,torch

class Manage:
    def __init__(self):
        self.sr_model_manage_dir = './Model'            # 数据库根路径
        self.sr_info_name = 'Info.json'

        self.sr_model_dc_temp_path = '{}/Model.pt'.format(self.sr_model_manage_dir)   # 模型参数临时保存路径,需要与Frame保持一致
        self.sr_model_py_temp_path = '{}/Model.py'.format(self.sr_model_manage_dir)   # 模型脚本临时保存路径,需要与Frame保持一致

        self.sr_data_path = '{}/Manage.json'.format(self.sr_model_manage_dir)       # 主文件路径
        self.dc_manage = None                           # 数据库主文件
        self.load_manage_json()                         # 加载 数据库主文件

    # 加载 数据库主文件
    def load_manage_json(self):
        with open(self.sr_data_path) as file_handel:
            self.dc_manage = json.load(file_handel)

    # 判断是否存在模型
    def exist_model_key_name(self,sr_model_key_name):
        if sr_model_key_name not in self.dc_manage:
            return False
        else:
            return True

    # 更新主文件
    def updata_dc_manage(self):
        with open(self.sr_data_path,'w') as file_handel:
            json.dump(self.dc_manage,file_handel)

    # 添加模型
    def add_model_item(self,sr_model_key_name):
        if sr_model_key_name not in self.dc_manage:
            self.dc_manage[sr_model_key_name] = {}
            self.updata_dc_manage()                 # 更新 主文件

            # 创建目录
            sr_model_dir = '{}/{}'.format(self.sr_model_manage_dir, sr_model_key_name)  # 模型文件信息目录
            os.mkdir(sr_model_dir)

            # 将模型py脚本复制到目录中
            sr_model_py_save_path = '{}/Model.py'.format(sr_model_dir,sr_model_key_name)   # 模型py文件路径
            shutil.copy(self.sr_model_py_temp_path, sr_model_py_save_path)  # 增加 模型py文件

            # 模型dc参数文件
            sr_model_dc_save_path = '{}/Model.pt'.format(sr_model_dir, sr_model_key_name)  # 模型dc文件路径

            sr_exec = sr_model_py_save_path.replace('.py', '').replace('.', '').replace('/', '.')[1:]
            sr_exec = 'from {} import {}'.format(sr_exec, sr_model_key_name)
            exec(sr_exec)                                           # 动态导入模型类,如 from Model.CNN.Model import CNN
            model = eval('{}()'.format(sr_model_key_name))          # 模型实例化, model = CNN()
            torch.save(model.state_dict(),sr_model_dc_save_path)    # 保存模型

            sr_model_info_path = '{}/{}'.format(sr_model_dir, self.sr_info_name)  # 模型信息路径
            # 添加 模型信息
            dc_info_json = {'train_log': []}
            with open(sr_model_info_path,'w') as file_handle:
                json.dump(dc_info_json, file_handle)

            return True
        else:
            raise Exception('模型 {} 已存在'.format(sr_model_key_name))

    # 更新模型
    def update_model_item(self,sr_model_key_name,dc_model):
        if sr_model_key_name not in self.dc_manage:
            raise Exception('模型 {} 不存在'.format(sr_model_key_name))
        else:
            sr_model_dir = '{}/{}'.format(self.sr_model_manage_dir, sr_model_key_name)  # 模型文件信息目录
            sr_model_dc_save_path = '{}/Model.pt'.format(sr_model_dir)                  # 模型文件路径
            sr_model_info_path = '{}/info.json'.format(sr_model_dir)                    # 模型信息路径

            # 更新 模型dc参数文件
            shutil.copy(self.sr_model_dc_temp_path, sr_model_dc_save_path)

            # 更新 模型信息
            dc_train_log = dc_model['train_log']
            dc_train_info = {
                'lr': dc_train_log['lr'],
                'epochs': dc_train_log['epochs'],
                'lossf': dc_train_log['lossf'],
                'optim': dc_train_log['optim'],
                'momentum': dc_train_log['momentum'],
                'time': dc_train_log['time'],
                'loss': dc_train_log['loss']
            }
            with open(sr_model_info_path) as file_handle:
                dc_info_json = json.load(file_handle)
            dc_info_json['train_log'].append(dc_train_info)
            with open(sr_model_info_path,'w') as file_handle:
                json.dump(dc_info_json,file_handle)
            return True

    # 删除模型
    def delete_model_item(self,sr_model_key_name):
        if sr_model_key_name in self.dc_manage:
            sr_model_dir = '{}/{}'.format(self.sr_model_manage_dir, sr_model_key_name)  # 模型文件信息目录
            shutil.rmtree(sr_model_dir)             # 删除 模型目录
            self.dc_manage.pop(sr_model_key_name)   # 删除 在数据库主文件中的模型键值
            self.updata_dc_manage()                 # 更新 数据库主文件
            return True
        else:
            print('>>> 不存在该模型.')
            return False

    # 查看模型
    def check_model_item(self,sr_model_key_name,mode=2):
        sr_model_dir = '{}/{}'.format(self.sr_model_manage_dir, sr_model_key_name)      # 模型文件信息目录
        sr_model_json_path = '{}/{}'.format(sr_model_dir,self.sr_info_name)             # 模型信息路径

        dc = {
            'model': None,
            'info': None
        }
        if mode == 0:
            # 只导入模型
            model = self.load_model(sr_model_key_name)
            dc['model'] = model
        elif mode == 1:
            # 只导入模型信息
            with open(sr_model_json_path) as file_handle:
                dc_info_json = json.load(file_handle)
            dc['info'] = dc_info_json
        elif mode == 2:
            # 导入模型和模型信息
            model = self.load_model(sr_model_key_name)
            with open(sr_model_json_path) as file_handle:
                dc_info_json = json.load(file_handle)
            dc['model'] = model
            dc['info'] = dc_info_json
        return dc

    # 根据模型键值名加载模型
    def load_model(self,sr_model_key_name):
        sr_model_dir = '{}/{}'.format(self.sr_model_manage_dir, sr_model_key_name)  # 模型文件信息目录
        sr_model_py_save_path = '{}/Model.py'.format(sr_model_dir, sr_model_key_name)  # 模型py文件路径

        # 模型dc参数文件
        sr_model_dc_save_path = '{}/Model.pt'.format(sr_model_dir, sr_model_key_name)  # 模型dc文件路径

        sr_exec = sr_model_py_save_path.replace('.py', '').replace('.', '').replace('/', '.')[1:]
        sr_exec = 'from {} import {}'.format(sr_exec, sr_model_key_name)
        exec(sr_exec)
        model = eval('{}()'.format(sr_model_key_name))

        model.load_state_dict(torch.load(sr_model_dc_save_path))

        return model








