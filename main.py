import pandas as pd
from catboost import CatBoostClassifier
from temp import data_preproccesing, get_db_connection


def add_data_to_db(df, connection):
    for i in df['index'].values:
        connection.execute("""
        UPDATE public."Test"
         SET "Probability" = {value}
        WHERE index={id}
        """.format(value=df.loc[df["index"] == i, "Probability"].values[0], id=i))


query = """
            SELECT * 
             FROM public."Test" 
            WHERE "Probability" IS NULL
             LIMIT 100
        """


if __name__ == '__main__':
    connect = get_db_connection()
    df = pd.read_sql(query, connect)
    X, y = data_preproccesing(df)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model('./model')
    preds = loaded_model.predict_proba(X)[:, 1]
    print(len(preds))
    df['Probability'] = preds
    add_data_to_db(df, connect)






