from .Model_Frame import Frame

def show_loss(ls_CNN,is_log=False):
    import matplotlib.pyplot as p
    import numpy as n

    fr = Frame()

    ls_legend = []
    for CNN_name,CNN_legend,CNN_color in ls_CNN:
        ls_legend.append(CNN_legend)
        fr.load_model(CNN_name)
        dc_info = fr.get_model_all_info()
        ls_loss = []
        for dc in dc_info['train_log']:
            ls_loss += dc['loss']
        if is_log:
            ls_loss = n.log10(ls_loss)
        if CNN_color is not None:
            p.plot(ls_loss,c=CNN_color)
        else:
            p.plot(ls_loss)

    p.xlabel('epoch')
    p.ylabel('loss')
    p.legend(ls_legend)
    p.grid()
    p.show()













