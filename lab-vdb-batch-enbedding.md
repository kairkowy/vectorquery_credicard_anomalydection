```python
# 저장된 keras 모델를 로딩하여 임베딩을 위한 인코더 부분만 분리하여 별도 다른 모델로 생성함.
# 별도 생성된 인코너 모델을 이용하여 테이블 데이터를 임베딩하는 시나리오임.
# 이작업은 jupyter notebooks의 메모리 부족 이슈가 발샐함으로 서버 사이드에서 배치로 돌리는 것이 좋을 수 있습니다.

import pandas as pd
import numpy as np
import keras
from keras.models import load_model, save_model
import array
import os
import oracledb
from sklearn import datasets, decomposition, preprocessing, model_selection

#uname = os.getenv("PYTHON_USERNAME")
#pw = os.getenv("PYTHON_PASSWORD")
#cs = os.getenv("PYTHON_CONNECTSTRING")

# DB 접속 및 커서 생성
uname = "vector"
pwd = "vector"
cns = "localhost:1521/freepdb1"

oracledb.init_oracle_client()
connection = oracledb.connect(user=uname, password=pwd, dsn=cns)
cursor = connection.cursor()
print("Connected to Oracle Database 23.4")
```

    2024-12-27 08:05:25.607564: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-12-27 08:05:25.620252: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-12-27 08:05:25.623897: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-12-27 08:05:25.633643: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-12-27 08:05:26.331318: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


    Connected to Oracle Database 23.4



```python
# Create model with latent space layer

CardFraudDector = load_model('./binds/creditcard-abnormal-model-db.keras', compile=False)
#CardFraudDector.summary()

CardFraudDectorEncode = keras.Model(
    inputs=CardFraudDector.inputs,
    outputs=CardFraudDector.get_layer(name="encoder").output,
)
CardFraudDectorEncode.summary()
print("complete to load a model")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_10"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">29</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ encoder (<span style="color: #0087ff; text-decoration-color: #0087ff">Sequential</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">865</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">865</span> (3.38 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">865</span> (3.38 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    complete to load a model

```python
# create a vector table
cr_table = """create table if not exist creditcardtr_vec 
              (id number, v vector, constraint fk_id foreign key(id) references creditcardtr(id))"""
cursor.execute(cr_table)
print("Cteated creditcardtr_vec table")

```python
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

#q_t = """select id, v1, v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
       v21,v22,v23,v24,v25,v26,v27,v28, amount, class from creditcardtr fetch first 10 rows only"""
#insert_v = """insert into creditcardtr_vec(id,v) values(:1,:2)"""
row_embedding = "update creditcardtr set v = :2 where id = :1 """

q = [q1,q2,q3,q4,q5,q6]
#q = [q_t]
import tensorflow as tf
for item in q:
    cursor.execute(item)
    rows = cursor.fetchall()
    col_name = cursor.description
    columns=[]
    for col in col_name:
        columns.append(col[0])

    df = pd.DataFrame(rows, columns=columns)
    #print(df)
    embeddingiRow = df[['ID']]
    embeddingdf = df.drop(labels=['ID','CLASS'], axis=1)
    embeddingdf = embeddingdf.to_numpy()
    # scaling
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(embeddingdf)
    scaled_data = scaler.transform(embeddingdf)
    #print(scaled_data)
    x = scaled_data
    #print(x)
    n_features = x.shape[1]
    rows = len(x)
    binds = []
    for i in range(rows):
        # extrect key
        key = int(embeddingiRow.iloc[i,0])
        # Create the embedding as single row and extract the vector
        inputrows_x = x[i].reshape(1,n_features)
        #encoded = CardFraudDectorEncode(inputrows_x)
        #encoded = tf.dtypes.cast(encoded,tf.float64)
        encoded = CardFraudDector(inputrows_x)
        encoded = tf.dtypes.cast(encoded,tf.float64)
        # Convert to list
        #print(encoded)
        vec = list(encoded[0])
        # Convert to array format
        vec2 = array.array("f", vec)
        binds.append([vec2,key])
        #print(key, vec2)
    print("Binds:",binds)
    cursor.executemany(row_embedding,binds)
    connection.commit()
    print("Embedding completes")
print("Finished Batch embedding")
connection.close()
```

# batch embedding for python code

```python
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
```