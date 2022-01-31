import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.feature_selection import VarianceThreshold as VT
import warnings
warnings.filterwarnings("ignore")

def scale(x):
    x = x.loc[:].copy()
    return (x - x.mean())/(x.std())
def fill_missing(x, Mode = False):
    if not Mode:
        if x.skew() > 0.5:
            x[x.isnull()] = x.median()
        else:
            x[x.isnull()] = x.mean()
    else:
        x[x.isnull()] = x.mode()[0]
    return x

train_data = pd.read_csv('train.csv')
data = train_data.drop(['Id','SalePrice','MoSold'], axis = 1)

data.MSZoning = data.MSZoning.fillna(data.MSZoning.mode()[0])
group = data.groupby(['MSZoning', 'Neighborhood'])['LotFrontage']
data['LotFrontage'] = group.transform(lambda x: x.fillna(x.mean()))

objCols = data.loc[:, data.select_dtypes('object').columns].copy()
numCols = data.loc[:, data.select_dtypes(['int64', 'float64']).columns].copy()

nan_to = 0
poor_to_good_map = {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1, np.nan: nan_to}
mask = objCols.apply(lambda col: col.str.contains('^Fa$|^TA$').any())
poor_to_good_cols = objCols.loc[:, mask]
ordinal_cols = poor_to_good_cols.apply(lambda col: col.map(poor_to_good_map))
objCols.drop(poor_to_good_cols, axis=1, inplace=True)

BsmtExposure_map = {'Gd': 4,'Av': 3,'Mn': 2,'No': 1, np.nan: nan_to}
ordinal_cols['BsmtExposure'] = objCols.pop('BsmtExposure').map(BsmtExposure_map)
BsntFinType_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, np.nan: nan_to}
BsntFinType_cols = objCols[['BsmtFinType1', 'BsmtFinType2']]
ordinal_cols[['BsmtFinType1', 'BsmtFinType2']] = BsntFinType_cols.apply(lambda x: x.map(BsntFinType_map))
objCols.drop(['BsmtFinType1', 'BsmtFinType2'], axis = 1, inplace = True)
Functional_map = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1, np.nan: nan_to}
ordinal_cols['Functional'] = objCols.pop('Functional').map(Functional_map)

GaraheFinish_map = {'Fin': 3, 'FRn': 2, 'Unf': 1, np.nan: nan_to}
ordinal_cols['GarageFinish'] = objCols.pop('GarageFinish').map(GaraheFinish_map)

mask = numCols.columns.str.contains('^Year|Yr')
years_cols = numCols.loc[:,mask].copy()
numCols[years_cols.columns] = years_cols.apply(lambda x: 2021 - x)

numCols = numCols.apply(fill_missing)
ordinal_cols = ordinal_cols.apply(fill_missing, Mode=True)
objCols.fillna('nan', inplace=True)

enc = OHE(sparse=False)
objCols = pd.DataFrame(enc.fit_transform(objCols[objCols.columns]))

all_cols = pd.concat([numCols, ordinal_cols, objCols], axis = 1)

sel = VT(threshold = 0.02)
sel.fit(all_cols)
all_cols = all_cols.loc[:, sel.get_support()]

cols = set(all_cols).intersection(set(numCols))
all_cols.loc[:, cols] = all_cols.loc[:, cols].apply(scale)

all_cols['GrLivAreaEXP'] = np.exp(all_cols['GrLivArea'])

np.save('processed.npy', all_cols)