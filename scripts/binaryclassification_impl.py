from datetime import datetime

import matplotlib
matplotlib.use("Agg")   

import matplotlib.pyplot as plt

from neonunicorn.deepl import binary_classification
from datetime import datetime
import matplotlib.pyplot as plt

from neonunicorn.deepl import binary_classification


def main():

    d = 10
    n = 2000

    epochs = 10000
    h = 0.001


    W1, W2, W3, W4, loss_history = binary_classification(d, n, epochs=epochs, h=h)
 
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Cross Entropy Loss vs Epoch")
    plt.tight_layout()


    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    pdf_name = f"crossentropyloss_{ts}.pdf"
    plt.savefig(pdf_name)

    print("Saved:", pdf_name)


if __name__ == "__main__":
    main()

