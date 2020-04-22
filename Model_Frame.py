from Model_Train import Train
from Model_Manage import Manage
from Need.Model_Predict import Predict
import torch,shutil


class Frame:
    '''
    #
    #
    #
    #
    #
        - 框架描述
            - 该框架启动后将把 Model_From.py 和 Model_Loader.py 复制到模型目录中并调用
            - 需要确保 Model_From.py 和 Model_Loader.py 的正确性


        - 使用时需要修改的文件:
            - MAIN.py               # 主文件
            - Need/Model_From.py    # 模型文件
            - Need/Model_Loader.py  # 训练集加载器
            - Need/Model_Predict.py # 模型预测(未完成)


        -  注意:
            - 以下目录名和文件名
                - Project
                    - Model                 # (不得修改目录名) 数据库目录
                    - Need                  # (不得修改目录名) 支持目录
                        - Model_From.py     # (不得修改文件名) 用于生成模型,必须包含模型实现类,模型的键值名称即类名
                        - Model_Loader.py   # (不得修改文件名) 用于返回 loader 训练集,必须包含 get_loader() 方法
                        - Model_Predict.py  # (不得修改文件名) 用于预测,必须包含 predict() 方法
                    - MAIN.py               # (可以修改文件名) 用户逻辑程序
                    - Model_Frame.py        # (不得修改文件名) 框架脚本
                    - Model_Manage.py       # (不得修改文件名) 数据库脚本
                    - Model_Train.py        # (不得修改文件名) 训练脚本



        - 使用示例:

            # MAIN.py

            fr = Frame()
            fr.load_model('FCNN')
            fr.set_train_params(
                lr=1e-3,
                epochs=10,
                lossf='mse',
                opt='adam'
            )
            fr.train()



    #
    #
    #
    #
    #
    '''



    def __init__(self):
        sr_tip = '\n\n' \
                 'Need目录在必须存在以下文件\n' \
                 '  - Model_From.py     用于生成模型,必须包含模型实现类,模型的键值名称即类名\n' \
                 '  - Model_Loader.py   用于返回 loader 训练集,必须包含 get_loader() 方法\n' \
                 '  - Model_Predict.py  用于预测,必须包含 predict() 方法' \
                 '\n\n'
        print(sr_tip)

        self.sr_model_manage_dir = './Model'

        self.sr_model_dc_temp_path = '{}/Model.pt'.format(self.sr_model_manage_dir)  # 模型参数临时保存路径,需要与Frame保持一致
        self.sr_model_py_temp_path = '{}/Model.py'.format(self.sr_model_manage_dir)  # 模型脚本临时保存路径,需要与Frame保持一致
        self.sr_loader_py_temp_path = '{}/Loader.py'.format(self.sr_model_manage_dir)  # 加载器脚本临时保存路径,需要与Frame保持一致

        self.sr_model_py_build_path = './Need/Model_From.py'   # 新模型文件 路径
        self.sr_loader_py_build_path = './Need/Model_Loader.py'

        self.model = None           # 模型实例，由 build,load 加载
        self.model_key_name = None  # 模型键值命名
        self.loader = None          # 训练数据集加载器
        self.dc_train_params = None # 训练参数, 由 set_train_params() 获得

        self.ins_Train = Train()    # 训练类实例
        self.ins_Manage = Manage()  # 管理类实例

    def init(self):
        self.model = None           # 模型实例，由 build,load 加载
        self.model_key_name = None  # 模型键值命名
        self.loader = None          # 训练数据集加载器
        self.dc_train_params = None # 训练参数, 由 set_train_params() 获得

    # 新建模型
    def build_model(self,sr_model_key_name):
        if not self.ins_Manage.exist_model_key_name(sr_model_key_name):
            print('>>> 新建模型 {}.'.format(sr_model_key_name))
            # 初始化变量
            self.init()
            try:
                # 将模型脚本临时保存
                self.save_temp_model_py()
                # 将加载器脚本临时保存
                self.save_temp_loader_py()
                # 添加模型
                self.ins_Manage.add_model_item(sr_model_key_name=sr_model_key_name)
                # 导入模型和加载器,从新添加的模型导入
                dc = self.ins_Manage.check_model_item(sr_model_key_name,0)
                self.model = dc['model']
                self.loader = dc['loader']
                # 模型键值名
                self.model_key_name =sr_model_key_name
            except:
                self.delete_model(sr_model_key_name)
                raise Exception('新建模型失败.')
        else:
            raise Exception('该模型已存在.')

    # 加载数据库中已存在的模型
    def load_model(self,sr_model_key_name):
        if self.ins_Manage.exist_model_key_name(sr_model_key_name):         # 数据库是否已存在该模型
            print('>>> 导入模型 {}.'.format(sr_model_key_name))
            # 初始化变量
            self.init()
            # 导入模型和加载器
            dc = self.ins_Manage.check_model_item(sr_model_key_name,mode=0)
            self.model = dc['model']
            self.loader = dc['loader']
            # 模型键值名
            self.model_key_name = sr_model_key_name
        else:
            raise Exception('该模型不存在.')

    def update_loader_py(self,sr_model_key_name):
        self.save_temp_loader_py()
        self.ins_Manage.update_loader_py(sr_model_key_name)

    # 模型预测
    def predict(self,dc_input):
        predict = Predict().predict(self.model,dc_input)
        return predict

    # 设置训练参数
    def set_train_params(self,lr,epochs,lossf,opt,momentum=0.9):
        self.dc_train_params = {
            'lr':lr,
            'epochs':epochs,
            'lossf':lossf,
            'optim':opt,
            'momentum':momentum,
        }

    # 获取模型的所有信息
    def get_model_all_info(self):
        dc = self.ins_Manage.check_model_item(self.model_key_name,mode=1)       # 调用管理实例获取模型信息
        dc_info = dc['info']
        return dc_info

    # 临时保存模型文件
    def save_temp_model_dc(self):
        torch.save(self.model.state_dict(), self.sr_model_dc_temp_path)     # 保存到模型临时存储
    # 临时保存模型脚本
    def save_temp_model_py(self):
        shutil.copy(self.sr_model_py_build_path,self.sr_model_py_temp_path)
    # 临时保存加载器脚本
    def save_temp_loader_py(self):
        shutil.copy(self.sr_loader_py_build_path, self.sr_loader_py_temp_path)

    # 训练
    def train(self):
        model = self.model
        loader = self.loader
        dc_train_params = self.dc_train_params
        print('>>> 开始训练.')
        dc_model = self.ins_Train.train(model=model,loader=loader,dc_train_params=dc_train_params)  # 训练
        print('>>> 训练完成.')
        self.model = dc_model['model']      # 模型更新
        self.save_temp_model_dc()           # 保存
        self.ins_Manage.update_model_item(self.model_key_name,dc_model) # 数据库更新

    # 删除模型
    def delete_model(self,sr_model_key_name):
        self.init()
        is_succ = self.ins_Manage.delete_model_item(sr_model_key_name)        # 调用管理实例删除当前模型( 删除模型前必须加载模型 )
        if is_succ:
            print('>>> 删除 {} 成功.'.format(sr_model_key_name))
        else:
            print('>>> 删除 {} 失败.'.format(sr_model_key_name))




if __name__ == '__main__':

    # fr = Frame()
    # # fr.build_model('FCNN')
    # fr.load_model('FCNN')
    # fr.set_train_params(
    #     lr=1e-3,
    #     epochs=10,
    #     lossf='mse',
    #     opt='adam'
    # )
    # fr.train()

    # print(fr.get_model_all_info())

    # fr.delete_model('FCNN')

    pass











