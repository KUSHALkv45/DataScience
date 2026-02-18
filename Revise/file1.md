# MYSQL

# case statment 
```
select
case
when cond1 then result1
when cond2 then result2
.
...
else resultlast   end as NewColumn
from tableName

or use if statements
select if(cond1,result1,if(cond2,result2,..resultlast)) as NewColumn

ex :
select 
case 
when salary % 3 = 1 then "t1"
when salary % 3 = 1  then "t2"
else "t3" end as user_type

select
if(salary % 3 = 0,"t3",if(salary % 2 = 0,"t2" , "te")) as user_Type
```
# indexes:
```
CREATE INDEX index_name
ON table_name (column,…);

Drop an index:
DROP INDEX index_name;

CREATE UNIQUE INDEX index_name 
ON table_name (column,…);
```
# views
```
CREATE VIEW [IF NOT EXISTS] view_name 
AS Select query ;

CREATE OR REPLACE view_name 
AS 
select_statement;

DROP VIEW [IF EXISTS] view1, view2, …;
```

# joins

![All Joins](joinsVenn.png)

# searching

```
LIKE and RLIKE clauses are used to search desired records from the table.

The SQL LIKE clause compares a value to other values that are similar using wildcard operators.
With the LIKE operator, there are two wildcards that can be used.

The percentage sign (%)
The underscore (_)

The % sign can be used to indicate zero, one, or more characters. A single number or letter is represented by the underscore.
These symbols can be mixed and matched.

The syntax for LIKE clause:
SELECT select_list
FROM table_name
WHERE column LIKE ‘%pattern%’ (or ‘_ABC’);
E.g, ‘S%’ will fetch all values that start with S.
‘_AB’ will fetch all values that have A and B at second and third places respectively. 

In MySQL, this operator is used to pattern match a string expression against a pattern. 
Syntax:
SELECT select_list
FROM table_name
WHERE column RLIKE ‘regular_expression’;

```


