import rbfnn
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/data-latih.csv",';',thousands='.',decimal=',')
x_raw = rbfnn.preprocessing(df.PROUCTION)
y_raw = rbfnn.preprocessing(df.PANEN)

hasil = rbfnn.train(x_raw)

print("error :",hasil['error'])
plt.plot(y_raw, '-x')
plt.plot(hasil['output'], '--x')
plt.show()