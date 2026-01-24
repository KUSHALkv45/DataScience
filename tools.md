``` python


    customers.drop_duplicates(subset = "email",keep = "first",inplace = True)

    # subset - considers these col when given or it considers all columns
    # keep - which one to keep -> first , last , False - removes all


    


     df = students.dropna(axis = 0 , how = 'all' ,  subset = "name" , inplace = False)

    return df

    # axis - 0 - rows , 1- columns 
    # how - any means any one of the value of the row or all
    # subset - which col or rows to consider if we include this how may be not so useful
    # inplace - as per name



    students = students.rename(
        columns = {
            "id" : "student_id",
            "first" : "first_name",
            "last" : "last_name",
            "age" : "age_in_years"
        }
    )



    students = students.astype({"grade":int})


    products["quantity"].fillna(0,inplace = True)
    # products.fillna(value = {col_name : value} , inplace = True)
    

```
