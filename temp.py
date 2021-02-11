import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
# from sqlalchemy import create_engine


def get_db_connection():
    return create_engine(f'postgresql://postgres@localhost:5433/postgres')


def data_preproccesing(df):
    df.loc[df.TotalSpent.str.len() == 1, 'TotalSpent'] = \
        df.loc[df.TotalSpent.str.len() == 1, 'MonthlySpending'].map(lambda x: x)
    df.TotalSpent = df.TotalSpent.astype('float')
    return df


df = pd.read_csv('train.csv')
df = data_preproccesing(df)
train, test = train_test_split(df, stratify=df.Churn, random_state=17, shuffle=True, test_size=0.2)




X = df.drop('Churn', 1)
y = df.Churn.values

cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

model = CatBoostClassifier(iterations=161, random_seed=17, loss_function='Logloss')

model.fit(X, y, cat_features=cat_cols, silent=True)


model.save_model('./model')


conn = get_db_connection()
test.to_sql('Test', con=conn)

