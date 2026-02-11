# rank
- rank () over (partition by user_id , date(opn) order by opn desc)  as rank
- dense_rank ||

# lag
- LAG(expression, offset, default_value) OVER ([PARTITION BY partition_column, ...]ORDER BY order_column [ASC|DESC])
