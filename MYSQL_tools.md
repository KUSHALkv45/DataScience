```
# rank
- rank () over (partition by user_id , date(opn) order by opn desc)  as rank
- dense_rank ||

# lag
- LAG(expression, offset, default_value) OVER ([PARTITION BY partition_column, ...]ORDER BY order_column [ASC|DESC])

# date_format
- date_format(date , 'expression') , expression - %Y , %y , %M , %m , %D , %d


# rng total
- SELECT
    user_id,
    req,
    SUM(IF(req = 0, 1, 0)) OVER (
        PARTITION BY user_id
        ORDER BY visit_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM cte;

```
