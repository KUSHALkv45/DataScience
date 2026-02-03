``` python


    customers.drop_duplicates(subset = "email",keep = "first",inplace = True)

    # subset - considers these col when given or it considers all columns
    # keep - which one to keep -> first , last , False - removes all


    


    df = students.dropna(axis = 0 , how = 'all' ,  subset = "name" , inplace = False)
    df["salary"].dropna().nlargest(5)
    df["salary"].dropna().nsmallest(5)

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


    ★★★
    df["avg_salary"] = df.groupby("department")["salary"].transform("mean")


    df["rank"] = df.groupby("month")["amt"].rank(method = "min",ascending = False)

    rank() method	                Tie handling	                                                100, 90, 90, 80
    average (default)	            Tied values get the average of their ranks	                    1, 2.5, 2.5, 4
    min	                            Tied values get the best (smallest) rank	                    1, 2, 2, 4
    max	                            Tied values get the worst (largest) rank	                    1, 3, 3, 4
    first	                        Ranked by order of appearance	                                1, 2, 3, 4
    dense	                        Like min, but no gaps                                           1, 2, 2, 3

    df.loc[df["rank"] == 1][["month","description","amt"]].sort_values(by = "month")


    running sum : df["running_sum"] = df.groupby("id")["col1"].cumsum()
```
