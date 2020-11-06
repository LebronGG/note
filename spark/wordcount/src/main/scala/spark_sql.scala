import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row

object spark_df {
  def main(args: Array[String]) {
    val spark = SparkSession.builder.master("local").appName("spark_df").getOrCreate()
    //使支持RDDs转换为DataFrames及后续sql操作
    import spark.implicits._
    val df = spark.read.json("file:///home/hadoop/project/wordcount/person.json")
    df.select(df("name"),df("age")+1).show()
    //    df.select($"name", $"age" + 1).show()
    df.filter($"age">=19).show()
    df.filter(df("age") > 20 ).show()
    df.groupBy($"age">18).count.show()
    df.groupBy("age").count().show()
    df.sort(df("age").desc).show()
    df.sort(df("age").desc, df("name").asc).show()
    df.select(df("name").as("username"),df("age")).show()
  }
}

object spark_action {
  def main(args: Array[String]) {
    val spark = SparkSession.builder.master("local").appName("spark_df").getOrCreate()
    //使支持RDDs转换为DataFrames及后续sql操作
    import spark.implicits._
    val df = spark.read.json("file:///home/hadoop/project/wordcount/person.json")
//    df.select($"name", $"age" + 1).show()
    df.filter($"age">=19).show()
    df.groupBy($"age">18).count.show()
  }
}

object read_json {
  def main(args: Array[String]) {
    val spark = SparkSession.builder.master("local").appName("read_json").getOrCreate()
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    val df = spark.read.json("file:///home/hadoop/project/wordcount/person.json")
//    val df = spark.read.format("json").load("file:///home/hadoop/project/wordcount/person.json")
    df.createOrReplaceTempView("people")
    val sqlDF = spark.sql("SELECT * FROM people")
    df.show()
  }
}

object spark_dfs {
  def main(args: Array[String]) {
    val spark = SparkSession.builder.master("local").appName("spark_dfs").getOrCreate()
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    val df = spark.read.orc("hdfs:///hive/zxx.db/t").toDF("a","b")
    df.createOrReplaceTempView("date")
//    spark.sql("show tables").show()

    spark.sql("insert into table date select 4,'d'")
    spark.sql("select * from date").show()
  }
}


