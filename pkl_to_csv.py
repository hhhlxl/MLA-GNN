import pickle as pkl
import pandas as pd

with open(r'gbmlgg15cv_all_st_0_0_0.pkl', "rb") as f:
	object = pkl.load(f,encoding='utf-8')
df = pd.DataFrame(object)
df.to_csv(r'split1.csv')
