import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ids = np.random.randint(low=0, high=123456, size=2000, dtype=np.int64)
scores = np.random.randn(2000)

df = pd.DataFrame({"ids": ids, "scores": scores})

ids_a = pa.array(ids)
scores_a = pa.array(scores, type=pa.float32())

table = pa.table([ids_a, scores_a], ["ids", "scores"])
print(table)

pq.write_table(table, "table.parquet")
df.to_csv("table.csv", index=False)
