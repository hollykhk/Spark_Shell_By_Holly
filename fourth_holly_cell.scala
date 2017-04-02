import org.apache.spark.sql.functions._
val filePath = "/FileStore/tables/h2tqh7vb1491165896749/car_milage_csv-9d74c.1"
val cars = spark.read.option("header","true"). option("inferSchema","true").csv(filePath)
println("Cars has "+ cars.count() +" rows") 
// Show the top 5 cars in the dataset
cars.show(5) 

// Print the schema of the dataset
cars.printSchema()
// Let's run some statistical functions

// Find summary of the following columns - mpg,hp,weight,automatic
cars.select("mpg","hp","weight","automatic").show()

// Run the following query - What is the average mpg and torque of the automatic and non-automatic cars?
// org.apache.spark.sql.AnalysisException: cannot resolve '`mpg`' given input columns: [automatic];;
// cars.select("automatic").describe("mpg","torque").show()

// Find the overall average mpg and torque of all the cars together
cars.select(mean(cars("mpg")), mean(cars("torque")) ).show()

// Find the correlation coefficient between weight and hp for all cars. Display only 4 decimal points
val cor = cars.stat.corr("hp","weight") 
println("hp to weight : Correlation = %.4f".format(cor))
