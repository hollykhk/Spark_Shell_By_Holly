import org.apache.spark.sql.functions._
val filePath = "/FileStore/tables/0bbiuyxm1491172678670/titanic3_02.csv" 
val passengers = spark.read.option("header","true"). option("inferSchema","true"). csv(filePath) 
// How many passengers were there on titanic?
println("Passengers has "+ passengers.count() +" rows")
// Let's get the columns that we need further
val passengers1 = passengers.select(passengers("Pclass"),passengers("Survived"),passengers("Gender"),passengers("Age"),passengers("SibSp"),passengers("Parch"),passengers("Fare"))
// print the schema of passengers1
passengers1.printSchema()
// Let's run some queries
// Find the gender distribution of passengers
passengers1.alias("Gender").show() 
// We would like to find out the joint distribution of gender and survived columns i.e. what is the number of males and females that survived
passengers1.stat.crosstab("Survived","Gender").show() 
