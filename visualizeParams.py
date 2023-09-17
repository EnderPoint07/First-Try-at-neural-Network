import numpy as np
from PIL import Image

def main():
    with np.load('saved_model.npz') as model:
        wIH = model['wIH']
        bIH = model['bIH']
        wHO = model['wHO']
        bHO = model['bHO']

    img = Image.fromarray(wIH, 'L')
    img.show()
    img = Image.fromarray(bIH, 'L')
    img.show()
    img = Image.fromarray(wHO, 'L')
    img.show()
    img = Image.fromarray(bHO, 'L')
    img.show()




if __name__ == '__main__':
    main()
