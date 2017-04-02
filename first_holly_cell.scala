import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
val data = sc.textFile("/FileStore/tables/qt5qvaey1491163359132")
data.count()
// display contents of data RDD
data.collect().foreach(println)
// let's convert qualitative variables to quantitative ones
def getDoubleValue( input:String ) : Double = {
    var result:Double = 0.0
    if (input == "P")  result = 3.0 
    if (input == "A")  result = 2.0
    if (input == "N")  result = 1.0
    if (input == "NB") result = 1.0
    if (input == "B")  result = 0.0
    return result
   }
// note that the column at index 6 is the class label and the rest are features
val parsedData = data.map{line => 
    val parts = line.split(",")
    LabeledPoint(getDoubleValue(parts(6)), Vectors.dense(parts.slice(0,6).map(x => getDoubleValue(x))))
}
// display contents of parsedData RDD
parsedData.collect().foreach(println)
// another way
println(parsedData.take(10).mkString("\n"))
// for classification, you need training and test parts. You want to randomly split data into train and test
val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
val trainingData = splits(0)
val testData = splits(1)
// check the data
trainingData.take(10).foreach(println)
// train the model
val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)
// dataframe way to do it
val parsedData1 = data.map{line => 
     line.split(",").map(getDoubleValue).mkString(",")
}
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count
println("Training Error = " + trainErr)
