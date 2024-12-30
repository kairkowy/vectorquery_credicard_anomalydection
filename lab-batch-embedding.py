import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
from keras import layers, activations, losses, optimizers, metrics, models
from keras.models import load_model, save_model
import array
from sklearn import datasets, decomposition, preprocessing, model_selection

import oracledb
import os

import tensorflow as tf

# Create model with latent space layer

CardFraudDector = load_model('./binds/creditcard-abnormal-model-db.keras', compile=False)
#CardFraudDector.summary()

CardFraudDectorEncode = keras.Model(
    inputs=CardFraudDector.inputs,
    outputs=CardFraudDector.get_layer(name="encoder").output,
)

print("complete to load a model")
uname = os.getenv("PYTHON_USERNAME")
pw = os.getenv("PYTHON_PASSWORD")
cs = os.getenv("PYTHON_CONNECTSTRING")

# DB 접속 및 커서 생성
uname = "vector"
pwd = "vector"
cns = "localhost:1521/freepdb1"

oracledb.init_oracle_client()
connection = oracledb.connect(user=uname, password=pwd, dsn=cns)
cursor = connection.cursor()
print("Connected to Oracle Database 23.4")

# Get a data from the table, batch embedding it and then save it to the DB table.
# When the amount of data is large, splitting the data is recommended.

q1 = """select id, v1, v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
       v21,v22,v23,v24,v25,v26,v27,v28, amount, class from creditcardtr where id >=1 and id <= 50000"""
q2 = """select id, v1, v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
       v21,v22,v23,v24,v25,v26,v27,v28, amount, class from creditcardtr where id >=50001 and id <= 100000"""
q3 = """select id, v1, v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
       v21,v22,v23,v24,v25,v26,v27,v28, amount, class from creditcardtr where id >=100001 and id <= 150000"""
q4 = """select id, v1, v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
       v21,v22,v23,v24,v25,v26,v27,v28, amount, class from creditcardtr where id >=150001 and id <= 200000"""
q5 = """select id, v1, v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
       v21,v22,v23,v24,v25,v26,v27,v28, amount, class from creditcardtr  where id >=200001 and id <= 250000"""
q6 = """select id, v1, v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
       v21,v22,v23,v24,v25,v26,v27,v28, amount, class from creditcardtr where id >=250001 and id <= 300000"""

#insert_v = """insert into creditcardtr_vec(id,v) values(:1,:2)"""
row_embedding = "update creditcardtr set v = :1 where id = :2 """

q = [q1,q2,q3,q4,q5,q6]

for item in q:
    cursor.execute(item)
    row = cursor.fetchall()
    col_name = cursor.description
    columns=[]
    for col in col_name:
        columns.append(col[0])

    df = pd.DataFrame(row, columns=columns)

    embeddingiRow = df[['ID']]
    embeddingdf = df.drop(labels=['ID','CLASS'], axis=1).to_numpy()
    #embeddingdf = embeddingdf.to_numpy()
    #print("fetch rows:", len(embeddingiRow))
    # scaling
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(embeddingdf)
    scaled_data = scaler.transform(embeddingdf)
    #print(scaled_data)
    x = scaled_data
    #x = embeddingdf
    n_features = x.shape[1]
    rows = len(x)
    binds = []
    for i in range(rows):
        # Create the embedding as single row and extract the vector
        inputrows_x = x[i].reshape(1,n_features)
        key = int(embeddingiRow.iloc[i,0])
        encoded = CardFraudDectorEncode(inputrows_x)
        encoded = tf.dtypes.cast(encoded,tf.float64)
        # Convert to list
        vec = list(encoded[0])
        # Convert to array format
        vec2 = array.array("f", vec)
        #vector = array.array("f",embedding[0])
        binds.append([vec2,key])
        #print(key, vec2)
    #print("Binds:",binds)
    cursor.executemany(row_embedding,binds)
    connection.commit()
    print("Embedding completes")
print("Finished Batch embedding")
connection.close()

import datetime
now = datetime.datetime.now()
print("현재 날짜와 시간:", now)

