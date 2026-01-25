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
    df["age_bucket"] = pd.to_numeric(df["age_bucket"], errors='coerce').astype(int)


    products["quantity"].fillna(0,inplace = True)
    # products.fillna(value = {col_name : value} , inplace = True)
    
    df1 = pd.concat([df1,df2],axis = 0) , # axis = 1 -> concats horizontally i.e column wise


     ans = weather.pivot(index = "month",columns = "city",values = "temperature")
    # month_order = ["January", "February", "March", "April", "May", "June", "July", "August","September", "October", "November", "December"]
    # ans = ans.reindex(month_order)   - returns nulls for empty months
    return ans



    df = report.melt(
        id_vars = ["product"],
        value_vars = ["quarter_1","quarter_2","quarter_3","quarter_4"],
        var_name = "quarter",
        value_name = "sales"
    )
    return df


    animals[animals["weight"] > 100].sort_values(by = "weight",ascending = False)[["name"]]
```
