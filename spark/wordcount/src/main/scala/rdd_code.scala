import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object rdd_Array {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("rdd_Array").setMaster("local")
    val sc = new SparkContext(conf)
    val array = Array(1,2,3,4,5)
    val rdd = sc.parallelize(array)
    rdd.foreach(println)
  }
}
object rdd_List {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("rdd_List").setMaster("local")
    val sc = new SparkContext(conf)
    val list = List(2,3,4,5,6)
    val rdd = sc.parallelize(list)
    rdd.foreach(println)
  }
}

object rdd_text {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("rdd_text").setMaster("local")
    val sc = new SparkContext(conf)
    val list = List("Hadoop","Spark","Hive")
    val rdd = sc.parallelize(list)
    println(rdd.collect().mkString(","))
  }
}
//持久化

object rdd_cache {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("rdd_cache").setMaster("local")
    val sc = new SparkContext(conf)
    val list = List("Hadoop","Spark","Hive")
    val rdd = sc.parallelize(list)
    rdd.cache() //会调用persist(MEMORY_ONLY)，但是，语句执行到这里，并不会缓存rdd，这是rdd还没有被计算生成
    println(rdd.count()) //第一次行动操作，触发一次真正从头到尾的计算，这时才会执行上面的rdd.cache()，把这个rdd放到缓存中
    println(rdd.collect().mkString(",")) //第二次行动操作，不需要触发从头到尾的计算，只需要重复使用上面缓存中的rdd
//    最后，可以使用unpersist()方法手动地把持久化的RDD从缓存中移除。
  }
}

//分区

object rdd_partition {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("rdd_partition").setMaster("local")
    val sc = new SparkContext(conf)
    val list = List("Hadoop","Spark","Hive")
    val rdd = sc.parallelize(list,2)
    println(rdd.collect().mkString(","))
  }
}
//kv
object rdd_kv {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("rdd_kv").setMaster("local")
    val sc = new SparkContext(conf)
    val rdd = sc.parallelize(Array(("spark",2),("hadoop",6),("hadoop",4),("spark",6)))
    val end = rdd.mapValues(x => (x,1)).reduceByKey((x,y) => (x._1+y._1,x._2 + y._2)).mapValues(x => (x._1 / x._2))
    end.foreach(println)
  }
}