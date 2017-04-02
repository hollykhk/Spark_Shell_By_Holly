import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.corr
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.RegressionEvaluator
	val filePath = "/FileStore/tables/o1q3k3531491165543800/car_milage-6f50d.csv"
	val cars = spark.read.option("header","true"). option("inferSchema","true").csv(filePath)
	val cars1 = cars.na.drop() 
	val assembler = new VectorAssembler()
 assembler.setInputCols(Array("displacement","hp","torque","CRatio","RARatio","CarbBarrells","NoOfSpeed","length","width","weight","automatic"))
    assembler.setOutputCol("features")
    val cars2 = assembler.transform(cars1)
    cars2.show(40)
    val train = cars2.filter(cars1("weight") <= 4000)
    val test = cars2.filter(cars1("weight") > 4000)
    test.show()
    println("Train = "+train.count()+" Test = "+test.count())
		val algLR = new LinearRegression()
		algLR.setMaxIter(100)
		algLR.setRegParam(0.3)
		algLR.setElasticNetParam(0.8)
		algLR.setLabelCol("mpg")
		val mdlLR = algLR.fit(train)
		println(s"Coefficients: ${mdlLR.coefficients} Intercept: ${mdlLR.intercept}")
		val trSummary = mdlLR.summary
    println(s"numIterations: ${trSummary.totalIterations}")
    println(s"Iteration Summary History: ${trSummary.objectiveHistory.toList}")
    trSummary.residuals.show()
    println(s"RMSE: ${trSummary.rootMeanSquaredError}")
    println(s"r2: ${trSummary.r2}")
    val predictions = mdlLR.transform(test)
    predictions.show()
    val evaluator = new RegressionEvaluator()
		evaluator.setLabelCol("mpg")
		val rmse = evaluator.evaluate(predictions)
		println("Root Mean Squared Error = "+"%6.3f".format(rmse))
		evaluator.setMetricName("mse")
		val mse = evaluator.evaluate(predictions)
		println("Mean Squared Error = "+"%6.3f".format(mse))
    println("** That's All Folks **")	
