# MYSQL

- indexes:
```
CREATE INDEX index_name
ON table_name (column,…);

Drop an index:
DROP INDEX index_name;

CREATE UNIQUE INDEX index_name 
ON table_name (column,…);
```

- views
```
CREATE VIEW [IF NOT EXISTS] view_name 
AS Select query ;

CREATE OR REPLACE view_name 
AS 
select_statement;

DROP VIEW [IF EXISTS] view1, view2, …;
```




