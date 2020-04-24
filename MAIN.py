from Frame.Model_Frame import Frame



if __name__ == '__main__':
    '''
    需要修改的文件:
        - MAIN.py               # 主文件
        - Need/Model_From.py    # 模型文件
        - Need/Model_Loader.py  # 训练集加载器
        - Need/Model_Predict.py # 模型预测
    
        - 示例:
        fr = Frame()
        fr.load_model('FCNN')
        fr.set_train_params(
            lr=1e-3,
            epochs=10,
            lossf='mse',
            opt='adam',
            momentum=0.9,
            is_show_detail=False,
        )
        fr.train()
    '''

    help(Frame)

    pass









