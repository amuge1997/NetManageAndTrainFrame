from .Model_Frame import Frame

def show_loss(ls_CNN):
    import matplotlib.pyplot as p
    import numpy as n

    fr = Frame()

    for CNN in ls_CNN:
        fr.load_model(CNN)
        dc_info = fr.get_model_all_info()
        ls_loss = []
        for dc in dc_info['train_log']:
            ls_loss += dc['loss']
        ls_loss_log = n.log10(ls_loss)
        p.plot(ls_loss_log)
    p.legend(ls_CNN)
    p.grid()
    p.show()













