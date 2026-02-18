# MYSQL

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

The SQL LIKE clause compares a value to other values that are similar using wildcard operators. With the LIKE operator, there are two wildcards that can be used.

The percentage sign (%)
The underscore (_) 
The % sign can be used to indicate zero, one, or more characters. A single number or letter is represented by the underscore. These symbols can be mixed and matched.
The syntax for LIKE clause
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


